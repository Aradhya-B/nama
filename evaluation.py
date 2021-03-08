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
COLUMN_HEADER_1 = 'string0'
COLUMN_HEADER_2 = 'string1'
COLUMN_HEADER_3 = 'score'


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

    print(f"| LSI Evaluation for {savePath}")
    print("---------------------------------------")
    print(
        f'Matches for {savePath} predicted in {time.time() - predictStartTime} seconds.')

    iterationStartTime = time.time()
    for minScore in np.arange(0.01, 1.01, 0.01):
        matchesOverMinScore = suggestedMatches[suggestedMatches['score'] >= minScore]
        thisPredictedMatcher = _createPredictedMatcher(testMatchesDF)
        thisPredictedMatcher.addMatchDF(matchesOverMinScore)
        compareScore = scoreFunc(testMatcher, predictedMatcher)
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


def _createTrainMatcher():
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


def _createMatcherFromDF(DF):
    """Return matcher created from dataframe."""
    matcher = Matcher()
    matcher.addStrings(DF[COLUMN_HEADER_1])
    matcher.addStrings(DF[COLUMN_HEADER_2])
    matcher.addMatchDF(DF)
    return matcher
