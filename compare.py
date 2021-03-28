import numpy as np
from collections import Counter
import math
import networkx as nx
from itertools import product


def edge_compare(matcherA, matcherB):
    n = len(matcherA.strings(min_count=0))

    edges_A = {tuple(sorted((i, j))) for i, j in matcherA.G.edges()}
    edges_B = {tuple(sorted((i, j))) for i, j in matcherB.G.edges()}

    n_misses = len(edges_A ^ edges_B)

    return 1 - 2*n_misses/(n**2 - n)


def component_compare(matcherA, matcherB):

    n = len(matcherA.strings(min_count=0))

    components_A = [set(c) for c in matcherA.components()]
    components_B = [set(c) for c in matcherB.components()]

    n_misses = 0
    for c_A, c_B in list(product(components_A, components_B)):
        n_misses += len(c_A & c_B) * len(c_A ^ c_B)

    return round(abs(1 - 2*n_misses/(n**2 - n)), 3)

def confusion_matrix_by_component(matcher, gold_matcher):
    tp, fp = _compare_positives_by_component(matcher, gold_matcher)

    tp_g, fn = _compare_positives_by_component(gold_matcher, matcher)

    assert tp == tp_g

    n = len(matcher.strings())
    tn = 0.5 * n * (n - 1) - tp - fp - fn

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

def _compare_positives_by_component(matcher, gold_matcher, use_counts=True):
    """

    Computes the number of within-component string pairs that appear in:
    - Both the matcher and gold_matcher (True Positives)
    - The matcher, but not the gold matcher (False Positives)
    """

    gold_component_map = gold_matcher.componentMap()

    tp = 0
    fp = 0
    for c in matcher.components():
        if use_counts:
            gold_c_counts = Counter()
            for i in c:
                gold_c_counts[gold_component_map.get(i, -1)] += matcher.counts[i]
        else:
            gold_c_counts = Counter(gold_component_map.get(i, -1) for i in c)

        for gold_c, count in gold_c_counts.items():
            if gold_c != -1:
                # The counted strings are in a gold component
                """
                All pairs of strings that are in matcher component c, and
                gold_matcher component gold_c are True Positives.

                'count' gives the number of strings in this set, so there must be
                0.5*count*(count-1) unique (unordered) string pairs in this category.
                """

                tp += 0.5 * count * (count - 1)

                """
                All pairs of strings that are in matcher component c, with one
                string in the gold_matcher component gold_c and one string in
                a different gold_matcher component are False Positives.

                There must be count*(len(c)-count) unique pairs in this category

                We multiply this number by 0.5 to correct for the fact that each
                pair will be counted twice (once from each "end")
                """

                fp += 0.5 * count * (len(c) - count)

            else:
                # The counted strings are not in any gold component
                """
                When strings in the matcher are not in any gold matcher
                component, we treat them as if they are all in separate gold
                components(i.e., strings that are not in the gold matcher are
                assumed to not be matched to anything in the gold matcher).

                In this case, the strings pairs that would ordinarily be True
                Positives are instead False Positives.
                """

                fp += 0.5 * count * (count - 1)
                fp += 0.5 * count * (len(c) - count)

    return tp, fp


def _naive_confusion_matrix_by_component(matcher, gold_matcher):
    tp, tn, fp, fn = 0, 0, 0, 0
    for (matcher_string, gold_string) in product(matcher.strings(), gold_matcher.strings()):
        if matcher_string in nx.node_connected_component(gold_matcher.matches(), gold_string):
            if gold_string in nx.node_connected_component(matcher.matches(), matcher_string):
                tp += 1
            else:
                fn += 1
        else:
            if gold_string in nx.node_connected_component(matcher.matches(), matcher_string):
                fp += 1
            else:
                tn +=1

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def bp_compare(matcherA, matcherB):
    g1 = matcherA.G
    g2 = matcherB.G

    g1A = nx.to_numpy_matrix(g1)
    g2A = nx.to_numpy_matrix(g2)

    g1L = nx.laplacian_matrix(g1).todense()
    g2L = nx.laplacian_matrix(g2).todense()

    g1D = np.add(g1L, g1A)
    g2D = np.add(g2L, g2A)

    g1I = np.identity(len(g1))
    g2I = np.identity(len(g2))

    g1_HF = _compute_homophily_factor(g1D)
    g2_HF = _compute_homophily_factor(g2D)

    (g1a, g1c_prime) = _compute_FaBP_constants(g1_HF)
    (g2a, g2c_prime) = _compute_FaBP_constants(g2_HF)

    g1_epsilon = 1 / (1 + np.where(g1D == np.amax(g1D))[0][0])
    g2_epsilon = 1 / (1 + np.where(g2D == np.amax(g2D))[0][0])

    # g1_to_invert = np.add(g1I, g1a * g1D, -g1c_prime * g1A)
    # g2_to_invert = np.add(g2I, g2a * g2D, -g2c_prime * g2A)
    g1_to_invert = np.add(g1I, g1_epsilon**2 * g1D, -g1_epsilon * g1A)
    g2_to_invert = np.add(g2I, g2_epsilon**2 * g2D, -g2_epsilon * g2A)

    g1_inverse = np.linalg.inv(g1_to_invert)
    g2_inverse = np.linalg.inv(g2_to_invert)

    dist = np.linalg.norm(g1_inverse - g2_inverse)

    s = 1 / (1 + dist)

    return s


def _compute_homophily_factor(D):
    one_norm = 1 / (2 + 2 * np.where(D == np.amax(D))[0][0])
    c1 = 2 + np.sum(D)
    c2 = np.sum(np.square(D)) - 1
    frobenius_norm = math.sqrt(
        (-c1 + math.sqrt(c1**2 + 4 * c2)) / (8 * c2))
    return max(one_norm, frobenius_norm)


def _compute_FaBP_constants(HF):
    about_half_HF = HF - 0.5
    four_HF_squared = 4 * about_half_HF**2
    a = (four_HF_squared) / (1 - four_HF_squared)
    c_prime = (2 * about_half_HF) / (1 - four_HF_squared)
    return (a, c_prime)
