import unittest
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import src.cross_validation.cross_validation as cv 

class TestCrossValidation(unittest.TestCase):

    def setUp(self):

        iris_dataset = datasets.load_iris()

        self.X = iris_dataset.data[:, :3]
        self.y = iris_dataset.target

    def test_kfold(self):
        
        clf = RandomForestClassifier()
        X_df = pd.DataFrame(self.X)
        [cm, perf] = cv.kfold(clf, X_df, self.y, verbose=True)

        assert True

    def test_loo(self):
        
        clf = RandomForestClassifier()
        X_df = pd.DataFrame(self.X)
        [cm, perf] = cv.leave_one_out(clf, X_df, self.y, verbose=True)

        assert True