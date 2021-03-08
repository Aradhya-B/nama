import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    [corpHashEdgeScore, corpHashCompScore] = _evaluateCorpHash(testMatcher, testMatchesDF)
    [basicHashEdgeScore, basicHashComponentScore] = _evaluateBasicHash(testMatcher, testMatchesDF)
    
    print("Evaluated CorpHash and BasicHash.")
    print(f'Edge CorpHash: {corpHashEdgeScore}')
    print(f'Component CorpHash: {corpHashCompScore}')
    print(f'Edge BasicHash: {basicHashEdgeScore}')
    print(f'Component BasicHash: {basicHashComponentScore}')
    
    [trainMatcher, trainMatchesDF] = _createTrainMatcher()
    
    matchersAndDF = {
        'trainMatcher': trainMatcher,
        'testMatcher': testMatcher,
        'testMatchesDF': testMatchesDF, 
        'trainMatchesDF': trainMatchesDF,
    }

    _saveLSIPlot(compare.edge_compare, 'lsi_edge.png', matchersAndDF)
    _saveLSIPlot(compare.component_compare, 'lsi_component.png', matchersAndDF)
    _saveLSIPlot(compare.edge_compare, 'lsi_with_corp_edge.png', matchersAndDF, True)
    _saveLSIPlot(compare.component_compare,
                 'lsi_with_corp_component.png', matchersAndDF, True)


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


def _saveLSIPlot(scoreFunc, savePath, matchersAndDF, addCorpHash=False):
    trainMatcher = matchersAndDF['trainMatcher']
    testMatcher = matchersAndDF['testMatcher']
    trainMatchesDF = matchersAndDF['trainMatchesDF']
    testMatchesDF = matchersAndDF['testMatchesDF']

    lsi = LSIModel(trainMatcher)
    
    predictedMatcher = _createPredictedMatcher(testMatchesDF)
    if addCorpHash:
        predictedMatcher.matchHash(corpHash)
    suggestedMatches = predictedMatcher.suggestMatches(lsi)
    # TODO Determine where to create predicted matcher in this loop
    # Option 1
    for minScore in np.arange(0.01, 1.01, 0.01):
        # Option 2
        # predictedMatcher = _createPredictedMatcher(testMatchesDF)
        # Only need to create the match DF one time
        matchesOverMinScore = suggestedMatches[suggestedMatches['score'] >= minScore]
        thisPredictedMatcher = _createPredictedMatcher(testMatchesDF)
        thisPredictedMatcher.addMatchDF(matchesOverMinScore)

        compareScore = scoreFunc(testMatcher, predictedMatcher)
        plt.plot(minScore, compareScore, marker='o', markersize=3, color="red")
    plt.xlabel('Min Score')
    plt.ylabel('Compare Score')
    plt.show()
    plt.savefig(savePath)
    print(f"Plot saved to {savePath}.")
    plt.clf()


def _createTestMatcher():
    """Return test matcher as well as dataframe."""
    testMatchesDF = pd.read_csv(TEST_DATA_PATH, nrows=1000)
    testMatchesDF[COLUMN_HEADER_3] = 1

    testMatcher = _createMatcherFromDF(testMatchesDF)

    return [testMatcher, testMatchesDF]

def _createTrainMatcher():
    """Return training matcher as well as dataframe."""
    trainMatchesDF = pd.read_csv(TRAIN_DATA_PATH, nrows=1000)
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