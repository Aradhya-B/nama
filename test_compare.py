import unittest
from matcher import Matcher
import strings
from compare import confusion_matrix, confusion_matrix_slow
import pandas as pd
import os.path as path

TEST_DATA_PATH = path.abspath(__file__ + '/../old/trainingData/test.csv')

class TestCompare(unittest.TestCase):

    def setUp(self):
        gold_matches_df = pd.read_csv(TEST_DATA_PATH, nrows=100)
        # Each match in the gold matcher is assumed to be correct
        gold_matches_df['score'] = 1

        gold_matcher = Matcher()
        gold_matcher = gold_matcher.add(gold_matches_df['string0'])
        gold_matcher = gold_matcher.add(gold_matches_df['string1'])
        for row in gold_matches_df.to_numpy().tolist():
            gold_matcher = gold_matcher.unite([row[0], row[1]])

        predicted_matcher = Matcher()
        predicted_matcher = predicted_matcher.add(gold_matches_df['string0'])
        predicted_matcher = predicted_matcher.add(gold_matches_df['string1'])

        self.gold_matcher = gold_matcher
        self.predicted_matcher = predicted_matcher

    def test_efficient_confusion_matrix(self):
        self.predicted_matcher = self.predicted_matcher.unite(strings.simplify_corp)
        naive_cm = confusion_matrix_slow(self.predicted_matcher, self.gold_matcher)
        efficient_cm = confusion_matrix(self.predicted_matcher, self.gold_matcher)
        print('Naive Confusion Matrix:', naive_cm)
        print('Efficient Confusion Matrix:', efficient_cm)

        self.assertDictEqual(
            naive_cm,
            efficient_cm
        )

if __name__ == '__main__':
    unittest.main()
