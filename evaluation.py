import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nama.matcher import Matcher
from nama.hashes import corpHash
from nama.lsi import LSIModel
import nama.compare as compare

dirname = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(dirname, 'trainingData/train.csv')
TEST_DATA_PATH = os.path.join(dirname, 'trainingData/test.csv')
COLUMN_HEADER_1 = 'string0'
COLUMN_HEADER_2 = 'string1'
COLUMN_HEADER_3 = 'score'


def evaluateMatchingMethods():
    [trainMatcher, testMatcher, testMatchesDF, trainMatchesDF] = _createMatchers()
    [corpHashEdgeScore, corpHashCompScore] = _evaluateCorpHash()
    print("Evaluated CorpHash.")
    print(f'Edge CorpHash: {corpHashEdgeScore}')
    print(f'Component CorpHash: {corpHashCompScore}')
    _saveLSIPlot(compare.edge_compare, 'lsi_edge.png')
    _saveLSIPlot(compare.component_compare, 'lsi_component.png')
    _saveLSIPlot(compare.edge_compare, 'lsi_with_corp_edge.png', True)
    _saveLSIPlot(compare.component_compare,
                 'lsi_with_corp_component.png', True)


def _evaluateCorpHash():
    [trainMatcher, testMatcher, testMatchesDF, trainMatchesDF] = _createMatchers()
    predictedMatcher = _createPredictedMatcher(testMatchesDF, testMatchesDF)
    predictedMatcher.matchHash(corpHash)
    edgeScore = compare.edge_compare(testMatcher, predictedMatcher)
    compScore = compare.component_compare(testMatcher, predictedMatcher)
    return [edgeScore, compScore]


def _saveLSIPlot(scoreFunc, savePath, addCorpHash=False):
    [trainMatcher, testMatcher, testMatchesDF, trainMatchesDF] = _createMatchers()
    lsi = LSIModel(trainMatcher)

    suggestedMatches = testMatcher.suggestMatches(lsi)
    # TODO Determine where to create predicted matcher in this loop
    # Option 1
    for minScore in np.arange(0.01, 1.01, 0.01):
        # Option 2
        # predictedMatcher = _createPredictedMatcher(testMatchesDF)
        # Only need to create the match DF one time
        tempMatchesDF = suggestedMatches[suggestedMatches['score'] >= minScore]
        predictedMatcher = _createPredictedMatcher(
            tempMatchesDF, testMatchesDF)
        # predictedMatcher.matchSimilar(lsi, min_score=minScore)
        if addCorpHash:
            predictedMatcher.matchHash(corpHash)
        compareScore = scoreFunc(testMatcher, predictedMatcher)
        plt.plot(minScore, compareScore, marker='o', markersize=3, color="red")
    plt.xlabel('Min Score')
    plt.ylabel('Compare Score')
    plt.savefig(savePath)
    print(f"Plot saved to {savePath}.")
    plt.clf()


def _createMatchers():
    """Return training matcher and test matcher as well as their dataframes."""
    trainMatchesDF = pd.read_csv(TRAIN_DATA_PATH, nrows=1000)
    trainMatchesDF[COLUMN_HEADER_3] = 1
    testMatchesDF = pd.read_csv(TEST_DATA_PATH, nrows=1000)
    testMatchesDF[COLUMN_HEADER_3] = 1

    trainMatcher = _createMatcherFromDF(trainMatchesDF)
    testMatcher = _createMatcherFromDF(testMatchesDF)

    return [trainMatcher, testMatcher, testMatchesDF, trainMatchesDF]


# TODO: Predicted matcher should have all test strings, not just a temp df passed in this function
def _createPredictedMatcher(DF, baseline):
    """Return basic matcher setup to make predictions (without edges)."""
    predictedMatcher = Matcher()
    predictedMatcher.addStrings(baseline[COLUMN_HEADER_1])
    predictedMatcher.addStrings(baseline[COLUMN_HEADER_2])
    predictedMatcher.addMatchDF(DF)
    return predictedMatcher


def _createMatcherFromDF(DF):
    """Return matcher created from dataframe."""
    matcher = Matcher()
    matcher.addStrings(DF[COLUMN_HEADER_1])
    matcher.addStrings(DF[COLUMN_HEADER_2])
    matcher.addMatchDF(DF)
    return matcher
