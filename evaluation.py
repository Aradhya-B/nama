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


def evaluateMatchingMethods():
    [trainMatcher, testMatcher, predictedMatcher] = _createMatchers()
    [corpHashEdgeScore, corpHashCompScore] = _evaluateCorpHash()
    _saveLSIPlot(compare.edge_compare, 'lsi_edge.jpg')
    _saveLSIPlot(compare.component_compare, 'lsi_component.jpg')
    _saveLSIPlot(compare.edge_compare, 'lsi_with_corp_edge.jpg', True)
    _saveLSIPlot(compare.component_compare,
                 'lsi_with_corp_component.jpg', True)
    print(f'Edge CorpHash: {corpHashEdgeScore}')
    print(f'Component CorpHash: {corpHashCompScore}')


def _evaluateCorpHash():
    [trainMatcher, testMatcher, predictedMatcher] = _createMatchers()
    predictedMatcher.matchHash(corpHash)
    edgeScore = compare.edge_compare(testMatcher, predictedMatcher)
    compScore = compare.component_compare(testMatcher, predictedMatcher)
    return [edgeScore, compScore]


def _saveLSIPlot(scoreFunc, savePath, addCorpHash=False):
    [trainMatcher, testMatcher, predictedMatcher] = _createMatchers()
    if addCorpHash:
        predictedMatcher.matchHash(corpHash)
    lsi = LSIModel(trainMatcher)

    minScore = np.linspace(0, 1, 100)
    compareScore = scoreFunc(testMatcher,
                             predictedMatcher.matchSimilar(lsi,
                                                           min_score=minScore))

    plt.plot(minScore, compareScore)
    plt.xlabel('Min Score')
    plt.ylabel('Compare Score')
    plt.savefig(savePath)


def _createMatchers():
    """Return training matcher, test matcher, and prediction matcher."""
    trainMatchesDF = pd.read_csv(TRAIN_DATA_PATH)
    testMatchesDF = pd.read_csv(TEST_DATA_PATH)

    trainMatcher = _createMatcherFromDF(trainMatchesDF)
    testMatcher = _createMatcherFromDF(testMatchesDF)
    predictedMatcher = _createPredictedMatcher(testMatchesDF)

    return [trainMatcher, testMatcher, predictedMatcher]


def _createPredictedMatcher(DF):
    """Return basic matcher setup to make predictions (without edges)."""
    predictedMatcher = Matcher()
    predictedMatcher.addStrings(DF[COLUMN_HEADER_1])
    predictedMatcher.addStrings(DF[COLUMN_HEADER_2])
    return predictedMatcher


def _createMatcherFromDF(DF):
    """Return matcher created from dataframe."""
    matcher = Matcher()
    matcher.addStrings(DF[COLUMN_HEADER_1])
    matcher.addStrings(DF[COLUMN_HEADER_2])
    matcher.addMatchDF(DF)
    return matcher
