import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from nama.matcher import Matcher
from nama.hashes import corpHash
from nama.hashes import basicHash
from nama.lsi import LSIModel
import nama.compare as compare

dirname = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(dirname, 'trainingData/train.csv')
TEST_DATA_PATH = os.path.join(dirname, 'trainingData/test.csv')
FIGURE_OUTPUT_DIR = 'namaFigures'
COLUMN_HEADER_1 = 'string0'
COLUMN_HEADER_2 = 'string1'
COLUMN_HEADER_3 = 'score'


def evaluate_hash_from_test_data_path(
    scoring_func=compare.edge_compare,
    nrows=None,
    test_data_path=TEST_DATA_PATH,
    hash_type=basicHash,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
    ch_3=COLUMN_HEADER_3
):
    """

    Evaluates a given type of hash for a set of test data passed as a csv file path using a given scoring function.

    :param scoring_func: scoring function to be used for evaluation
    :type scoring_func: Callable
    :param nrows: number of rows to grab from csv
    :type nrows: int
    :param test_data_path: path to csv containing test data
    :type test_data_path: str
    :param hash_type: function of hash to evaluate (from nama.hashes module)
    :type hash_type: Callable
    :param ch_1: first column header in test csv
    :type ch_1: str
    :param ch_2: second column header in test csv
    :type ch_2: str
    :param ch_3: third column header in test csv
    :type ch_3: str
    :returns: computed score and time taken to generate matches as tuple
    """

    test_matches_df = _create_test_matches_df_from_test_data_path(
        test_data_path=test_data_path,
        nrows=nrows,
        ch_3=ch_3)
    return evaluate_hash_from_test_df(test_matches_df,
                                      scoring_func=scoring_func,
                                      hash_type=hash_type,
                                      ch_1=ch_1,
                                      ch_2=ch_2)


def evaluate_hash_from_test_df(
    test_matches_df,
    scoring_func=compare.edge_compare,
    hash_type=basicHash,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2
):
    """

    Evaluates a given type of hash for a set of test data passed as a pandas DataFrame using a given scoring function.

    :param test_matches_df: pandas df that holds gold standard matches
    :type test_matches_df: pandas.DataFrame
    :param scoring_func: scoring function to be used for evaluation
    :type scoring_func: Callable
    :param hash_type: function of hash to evaluate (from nama.hashes module)
    :type hash_type: Callable
    :param ch_1: first column header in test csv
    :type ch_1: str
    :param ch_2: second column header in test csv
    :type ch_2: str
    :returns: computed score and time taken to generate matches as tuple
    """
    test_matcher, predicted_matcher = _create_test_and_predicted_matcher(
        test_matches_df,
        ch_1=ch_1,
        ch_2=ch_2)
    match_start_time = time.time()
    predicted_matcher.matchHash(hash_type)
    match_end_time = time.time()
    return (scoring_func(test_matcher, predicted_matcher),
            round(match_end_time - match_start_time, 4))


def evaluate_trained_model_with_min_score_from_test_data_path(
    trained_model,
    saved_figure_name,
    nrows=None,
    test_data_path=TEST_DATA_PATH,
    scoring_func=compare.edge_compare,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
    ch_3=COLUMN_HEADER_3,
    add_corp_hash=False,
    show_plot=False
):
    """

    Evaluates a trained model (eg. lsi) for a set of test data passed as a csv file path using a given scoring function.

    :param trained_model: trained model to be evaluated eg. lsi
    :param saved_figure_name: name of figure (will be saved as name.png)
    :type saved_figure_name: str
    :param nrows: number of rows to grab from csv
    :type nrows: int
    :param test_data_path: path to csv containing test data
    :type test_data_path: str
    :param scoring_func: scoring function to be used for evaluation
    :type scoring_func: Callable
    :param ch_1: first column header in test csv
    :type ch_1: str
    :param ch_2: second column header in test csv
    :type ch_2: str
    :param ch_3: third column header in test csv
    :type ch_3: str
    :param add_corp_hash: should model be trained on corp hash as well
    :type add_corp_hash: boolean
    :param show_plot: should generated plots be shown at runtime
    :type show_plot: boolean
    :returns: max computed score, min filter score at max score, and time taken to predict matches using trained model
    """
    test_matches_df = _create_test_matches_df_from_test_data_path(
        test_data_path=test_data_path,
        nrows=nrows,
        ch_3=ch_3)
    return evaluate_trained_model_with_min_score_from_test_df(
        trained_model,
        test_matches_df,
        saved_figure_name,
        scoring_func=scoring_func,
        ch_1=ch_1,
        ch_2=ch_2,
        add_corp_hash=add_corp_hash,
        show_plot=show_plot)


def evaluate_trained_model_with_min_score_from_test_df(
    trained_model,
    test_matches_df,
    saved_figure_name,
    scoring_func=compare.edge_compare,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
    add_corp_hash=False,
    show_plot=False
):
    """

    Evaluates a trained model (eg. lsi) for a set of test data passed as a pandas DataFrame using a given scoring function.

    :param trained_model: trained model to be evaluated eg. lsi
    :param test_matches_df: pandas df that holds gold standard matches
    :type test_matches_df: pandas.DataFrame
    :param saved_figure_name: name of figure (will be saved as name.png)
    :type saved_figure_name: str
    :param scoring_func: scoring function to be used for evaluation
    :type scoring_func: Callable
    :param ch_1: first column header in test csv
    :type ch_1: str
    :param ch_2: second column header in test csv
    :type ch_2: str
    :param add_corp_hash: should model be trained on corp hash as well
    :type add_corp_hash: boolean
    :param show_plot: should generated plots be shown at runtime
    :type show_plot: boolean
    :returns: max computed score, min filter score at max score, and time taken to predict matches using trained model
    """
    test_matcher, predicted_matcher = _create_test_and_predicted_matcher(
        test_matches_df,
        ch_1=ch_1,
        ch_2=ch_2)
    match_start_time = time.time()
    if add_corp_hash:
        predicted_matcher.matchHash(corpHash)
    suggested_matches = predicted_matcher.suggestMatches(trained_model)
    match_end_time = time.time()

    max_score = float('-inf')
    min_score_at_max_score = 0.01
    for min_score in np.arange(0.01, 1.01, 0.01):
        matches_over_min_score = suggested_matches[
            suggested_matches['score'] >= min_score]
        this_predicted_matcher = _create_matcher_from_df(test_matches_df,
                                                         ch_1=ch_1,
                                                         ch_2=ch_2,
                                                         predicted=True)
        this_predicted_matcher.addMatchDF(matches_over_min_score)
        this_score = scoring_func(test_matcher, this_predicted_matcher)

        if this_score > max_score:
            max_score = this_score
            min_score_at_max_score = min_score

        plt.plot(min_score,
                 this_score,
                 marker='o',
                 markersize=3,
                 color='red')
    plt.xlabel('Min Score')
    plt.ylabel('Compare Score')
    if show_plot:
        plt.show()
    plt.savefig(f'{saved_figure_name}.png')
    return (max_score,
            min_score_at_max_score,
            round(match_end_time - match_start_time, 4))


def evaluate_all_matching_methods_from_test_data_path(
    trained_model,
    nrows=None,
    test_data_path=TEST_DATA_PATH,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
    ch_3=COLUMN_HEADER_3,
    show_plots=False
):
    """

    Print a comprehensive report to the command line after evaluating all possible scoring methods for a set of test data passed as a csv file path. Evaluates BasicHash, CorpHash, given trained model, and given trained model with CorpHash. Uses edge and component based scoring for all evaluations.

    :param trained_model: trained model to be evaluated eg. lsi
    :param nrows: number of rows to grab from csv
    :type nrows: int
    :param test_data_path: path to csv containing test data
    :type test_data_path: str
    :param ch_1: first column header in test csv
    :type ch_1: str
    :param ch_2: second column header in test csv
    :type ch_2: str
    :param ch_3: third column header in test csv
    :type ch_3: str
    :param show_plots: should generated plots be shown at runtime
    :type show_plots: boolean
    """
    basic_hash_edge_score, basic_hash_edge_time = \
        evaluate_hash_from_test_data_path(
            test_data_path=TEST_DATA_PATH,
            nrows=nrows,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3
        )
    basic_hash_component_score, basic_hash_component_time = \
        evaluate_hash_from_test_data_path(
            test_data_path=TEST_DATA_PATH,
            scoring_func=compare.component_compare,
            nrows=nrows,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3
        )
    corp_hash_edge_score, corp_hash_edge_time = \
        evaluate_hash_from_test_data_path(
            test_data_path=TEST_DATA_PATH,
            hash_type=corpHash,
            nrows=nrows,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3
        )
    corp_hash_component_score, corp_hash_component_time = \
        evaluate_hash_from_test_data_path(
            test_data_path=TEST_DATA_PATH,
            hash_type=corpHash,
            scoring_func=compare.component_compare,
            nrows=nrows,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3
        )
    print("| Evaluated CorpHash and BasicHash")
    print("----------------------------------")
    print(f'CorpHash Edge Score, Match Time: {corp_hash_edge_score}, {corp_hash_edge_time}')
    print(f'CorpHash Component Score, Match Time: {corp_hash_component_score}, {corp_hash_component_time}')
    print(f'BasicHash Edge Score, Match Time: {basic_hash_edge_score}, {basic_hash_edge_time}')
    print(f'CorpHash Component Score, Match Time: {basic_hash_component_score}, {basic_hash_component_time}')
    print('\n')

    tm_edge_score, tm_edge_min_at_max, tm_edge_time = \
        evaluate_trained_model_with_min_score_from_test_data_path(
            trained_model,
            'trained_edge_score',
            nrows=nrows,
            test_data_path=test_data_path,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3,
            show_plot=show_plots
        )

    tm_comp_score, tm_comp_min_at_max, tm_comp_time = \
        evaluate_trained_model_with_min_score_from_test_data_path(
            trained_model,
            'trained_comp_score',
            nrows=nrows,
            test_data_path=test_data_path,
            scoring_func=compare.component_compare,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3,
            show_plot=show_plots
        )

    tm_edge_with_corp_score,\
        tm_edge_with_corp_min_at_max,\
        tm_edge_with_corp_time = \
        evaluate_trained_model_with_min_score_from_test_data_path(
            trained_model,
            'trained_with_corp_edge_score',
            nrows=nrows,
            test_data_path=test_data_path,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3,
            add_corp_hash=True,
            show_plot=show_plots
        )

    tm_comp_with_corp_score,\
        tm_comp_with_corp_min_at_max,\
        tm_comp_with_corp_time = \
        evaluate_trained_model_with_min_score_from_test_data_path(
            trained_model,
            'trained_with_corp_comp_score',
            nrows=nrows,
            test_data_path=test_data_path,
            scoring_func=compare.component_compare,
            ch_1=ch_1,
            ch_2=ch_2,
            ch_3=ch_3,
            add_corp_hash=True,
            show_plot=show_plots
        )

    print("| Evaluated Trained Model")
    print("--------------------------")
    print(f'Edge Max Score, Min Filter Score for Max Score, Match Time: {tm_edge_score}, {tm_edge_min_at_max}, {tm_edge_time}')
    print(f'Component Max Score, Min Filter Score for Max Score Match Time: {tm_comp_score}, {tm_comp_min_at_max}, {tm_comp_time}')
    print('\n')

    print("| Evaluated Trained Model with CorpHash")
    print("---------------------------------------")
    print(f'Edge Max Score, Min Filter Score for Max Score, Match Time: {tm_edge_with_corp_score}, {tm_edge_with_corp_min_at_max}, {tm_edge_with_corp_time}')
    print(f'Component Max Score, Min Filter Score for Max Score Match Time: {tm_comp_with_corp_score}, {tm_comp_with_corp_min_at_max}, {tm_comp_with_corp_time}')
    print('\n')


def evaluateMatchingMethods():
    [testMatcher, testMatchesDF] = _createTestMatcher()
    [corpHashEdgeScore, corpHashCompScore] = _evaluateCorpHash(
        testMatcher, testMatchesDF)
    [basicHashEdgeScore, basicHashComponentScore] = _evaluateBasicHash(
        testMatcher, testMatchesDF)

    print("| Evaluated CorpHash and BasicHash")
    print("----------------------------------")
    print(f'Edge CorpHash Score: {corpHashEdgeScore}')
    print(f'Component CorpHash Score: {corpHashCompScore}')
    print(f'Edge BasicHash Score: {basicHashEdgeScore}')
    print(f'Component BasicHash Score: {basicHashComponentScore}\n')

    [trainMatcher, trainMatchesDF] = _createTrainMatcher()

    matchersAndDF = {
        'testMatcher': testMatcher,
        'testMatchesDF': testMatchesDF,
    }

    trainingStartTime = time.time()
    lsi = LSIModel(trainMatcher)
    print("| Completed LSI Training")
    print("--------------------------")
    print(f'LSI training time was {time.time() - trainingStartTime} seconds\n')

    _saveLSIPlot(compare.edge_compare, 'lsi_edge.png', matchersAndDF, lsi)
    _saveLSIPlot(compare.component_compare,
                 'lsi_component.png', matchersAndDF, lsi)
    _saveLSIPlot(compare.edge_compare,
                 'lsi_with_corp_edge.png', matchersAndDF, lsi, True)
    _saveLSIPlot(compare.component_compare,
                 'lsi_with_corp_component.png', matchersAndDF, lsi, True)


def _evaluateCorpHash(testMatcher, testMatchesDF):
    predictedMatcher = _createPredictedMatcher(testMatchesDF)
    predictedMatcher.matchHash(corpHash)

    edgeScore = compare.edge_compare(testMatcher, predictedMatcher)
    compScore = compare.component_compare(testMatcher, predictedMatcher)
    return [edgeScore, compScore]


def _evaluateBasicHash(testMatcher, testMatchesDF):
    predictedMatcher = _createPredictedMatcher(testMatchesDF)
    predictedMatcher.matchHash(basicHash)

    edgeScore = compare.edge_compare(testMatcher, predictedMatcher)
    compScore = compare.component_compare(testMatcher, predictedMatcher)
    return [edgeScore, compScore]


def _saveLSIPlot(scoreFunc, savePath, matchersAndDF, lsi, addCorpHash=False):
    testMatcher = matchersAndDF['testMatcher']
    testMatchesDF = matchersAndDF['testMatchesDF']

    predictStartTime = time.time()
    predictedMatcher = _createPredictedMatcher(testMatchesDF)
    if addCorpHash:
        predictedMatcher.matchHash(corpHash)
    suggestedMatches = predictedMatcher.suggestMatches(lsi)
    print(savePath, 'score distribution')
    print(suggestedMatches.describe())

    print(f"| LSI Evaluation for {savePath}")
    print("---------------------------------------")
    print(
        f'Matches for {savePath} predicted in {time.time() - predictStartTime} seconds.')

    iterationStartTime = time.time()
    for minScore in np.arange(0.01, 1.01, 0.01):
        matchesOverMinScore = suggestedMatches[suggestedMatches['score'] >= minScore]
        thisPredictedMatcher = _createPredictedMatcher(testMatchesDF)
        thisPredictedMatcher.addMatchDF(matchesOverMinScore)
        compareScore = scoreFunc(testMatcher, thisPredictedMatcher)
        plt.plot(minScore, compareScore, marker='o', markersize=3, color="red")

    plt.xlabel('Min Score')
    plt.ylabel('Compare Score')
    plt.show()
    plt.savefig(savePath)
    print(
        f"Plot saved to {savePath}. Min score iteration completed in {time.time() - iterationStartTime} seconds.\n")
    plt.clf()


def _createTestMatcher():
    """Return test matcher as well as dataframe."""
    testMatchesDF = pd.read_csv(TEST_DATA_PATH, nrows=100)
    testMatchesDF[COLUMN_HEADER_3] = 1

    testMatcher = _createMatcherFromDF(testMatchesDF)

    return [testMatcher, testMatchesDF]


def createTrainMatcher():
    """Return training matcher as well as dataframe."""
    trainMatchesDF = pd.read_csv(TRAIN_DATA_PATH, nrows=100)
    trainMatchesDF[COLUMN_HEADER_3] = 1

    trainMatcher = _createMatcherFromDF(trainMatchesDF)

    return [trainMatcher, trainMatchesDF]


# TODO: Predicted matcher should have all test strings, not just a temp df passed in this function
def _createPredictedMatcher(baseline):
    """Return basic matcher setup to make predictions (without edges)."""
    predictedMatcher = Matcher()
    predictedMatcher.addStrings(baseline[COLUMN_HEADER_1])
    predictedMatcher.addStrings(baseline[COLUMN_HEADER_2])
    return predictedMatcher


def _createMatcherFromDF(df):
    """Return matcher created from dataframe."""
    matcher = Matcher()
    matcher.addStrings(df[COLUMN_HEADER_1])
    matcher.addStrings(df[COLUMN_HEADER_2])
    matcher.addMatchDF(df)
    return matcher

# -------------------------------------------- util functions above not used


def _create_matcher_from_df(
    df,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
    predicted=False
):
    matcher = Matcher()
    matcher.addStrings(df[ch_1])
    matcher.addStrings(df[ch_2])
    if not predicted:
        matcher.addMatchDF(df)
    return matcher


def _create_test_matches_df_from_test_data_path(
    test_data_path=TEST_DATA_PATH,
    nrows=None,
    ch_3=COLUMN_HEADER_3
):
    test_matches_df = pd.read_csv(test_data_path, nrows=nrows)
    test_matches_df[ch_3] = ch_3
    return test_matches_df


def _create_test_and_predicted_matcher(
    test_matches_df,
    ch_1=COLUMN_HEADER_1,
    ch_2=COLUMN_HEADER_2,
):
    test_matcher = _create_matcher_from_df(test_matches_df,
                                           ch_1=ch_1,
                                           ch_2=ch_2)
    predicted_matcher = _create_matcher_from_df(test_matches_df,
                                                ch_1=ch_1,
                                                ch_2=ch_2,
                                                predicted=True)
    return (test_matcher, predicted_matcher)
