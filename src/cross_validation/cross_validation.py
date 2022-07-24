import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, balanced_accuracy_score
from src.performance.performance import PerformanceMetrics

# Put constants here

class CrossValidation():

    def __init__(self, X, y, ):
        self.X = X
        self.y = y
        return

    def kfold(self, model, k: int = 10, cv_name:str ='test', verbose: bool = False):
        
        k_fold = StratifiedKFold(n_splits = k)
        y_pred = pd.Series('NC', index = range(0, self.X.shape[0]))
        i_fold = 1

        print("k-fold cross-validation running...")

        for train_index, test_index in k_fold.split(self.X, self.y):

            # Defining training set and test set for the i-th fold
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Model training
            model.fit(X_train.values, y_train.values)
            y_pred[test_index] = model.predict(X_test.values)

            if(verbose):
                # Evaluationg performance of the i-th fold
                print("k = {i_fold}, balance acc = {1}, f1-score = {2}, MCC = {3}"
                    .format(i_fold, 
                            np.round(balanced_accuracy_score(y_test, y_pred[test_index]), decimals=2),
                            np.round(f1_score(y_test, y_pred[test_index], average= 'micro'), decimals=2),
                            np.round(matthews_corrcoef(y_test, y_pred[test_index]), decimals=2)))
            i_fold = i_fold + 1
        
        # Overall performances
        balanced_accuracy = np.round(balanced_accuracy_score(self.y, y_pred), decimals=2)
        f1score = np.round(f1_score(self.y, y_pred, average= 'micro'), decimals=2)
        mcc = np.round(matthews_corrcoef(self.y, y_pred), decimals=2)
        print("OVERALL, balanced acc = {0}, f1-score = {1}, MCC = {2}"
                .format(balanced_accuracy, f1score, mcc))
        conf_mat = confusion_matrix(self.y, y_pred)
        print("{0}-fold cross-validation classification report:".format(k))

        cv_perf = PerformanceMetrics(self.y, y_pred).classification_scores()

        df_temp = pd.DataFrame(pd.DataFrame(pd.Series([cv_name, balanced_accuracy, f1score, mcc])))
        df_cv_perf = df_temp.T
        df_cv_perf.columns = ['cv_type', 'balanced_accuracy', 'f1_score', 'mcc']

        return conf_mat, df_cv_perf

    def leave_one_out(self, model, cv_name: str ='test', verbose: bool = False):
        
        loo = LeaveOneOut()
        
        y_pred = pd.Series('NC', index = range(0, self.X.shape[0]))

        cnt = 1
        print("Leave One Out cross-validation...")
        for train_index, test_index in loo.split(self.X):

            # Defining training set and test set for the i-th fold
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Model training
            model.fit(X_train, y_train)
            y_pred[test_index] = model.predict(X_test)
            if(verbose):
                # Evaluationg performance of the i-th fold
                print("i = {0}, f1-score = {1}".format(cnt, np.round(f1_score(y_test, y_pred, average= 'micro'), decimals = 2)))
            cnt = cnt + 1
        
        # Overall performances
        conf_mat = confusion_matrix(self.y, y_pred)
        balanced_accuracy = np.round(balanced_accuracy_score(self.y, y_pred), decimals=2)
        f1score = np.round(f1_score(self.y, y_pred, average= 'micro'), decimals=2)
        mcc = np.round(matthews_corrcoef(self.y, y_pred), decimals=2)
        print("OVERALL, balanced acc = {0}, f1-score = {1}, MCC = {2}"
                .format(balanced_accuracy, f1score, mcc))
        conf_mat = confusion_matrix(self.y, y_pred)

        print("Leave One Out cross-validation classification report:")
        
        cv_perf = PerformanceMetrics(self.y, y_pred).classification_scores()

        df_temp = pd.DataFrame(pd.DataFrame(pd.Series([cv_name, balanced_accuracy, f1score, mcc])))
        df_cv_perf = df_temp.T
        df_cv_perf.columns = ['cv_type', 'balanced_accuracy', 'f1_score', 'mcc']

        return conf_mat, df_cv_perf

    def leave_one_subject_out(self, model, subject_ids: list, cv_name: str ='test', verbose: bool = False):

        y_pred = pd.Series('NC', index = range(0, self.X.shape[0]))

        subject_id = np.unique(subject_ids)

        print("Leave One Subject Out cross-validation...")

        for s in subject_id:

            test_index = subject_ids == s
            train_index = not(test_index)

            # Defining training set and test set for the i-th fold
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Model training
            model.fit(X_train, y_train)
            y_pred[test_index] = model.predict(X_test)
            if(verbose):
                # Evaluationg performance of the i-th fold
                print("i = {0}, f1-score = {1}".format(cnt, np.round(f1_score(y_test, y_pred, average= 'micro'), decimals = 2)))
            cnt = cnt + 1

        # Overall performances
        conf_mat = confusion_matrix(self.y, y_pred)
        balanced_accuracy = np.round(balanced_accuracy_score(self.y, y_pred), decimals=2)
        f1score = np.round(f1_score(self.y, y_pred, average= 'micro'), decimals=2)
        mcc = np.round(matthews_corrcoef(self.y, y_pred), decimals=2)
        print("OVERALL, balanced acc = {0}, f1-score = {1}, MCC = {2}"
                .format(balanced_accuracy, f1score, mcc))
        conf_mat = confusion_matrix(self.y, y_pred)

        print("Leave One Subject Out cross-validation classification report:")
        
        cv_perf = PerformanceMetrics(self.y, y_pred).classification_scores()

        df_temp = pd.DataFrame(pd.DataFrame(pd.Series([cv_name, balanced_accuracy, f1score, mcc])))
        df_cv_perf = df_temp.T
        df_cv_perf.columns = ['cv_type', 'balanced_accuracy', 'f1_score', 'mcc']

        return conf_mat, df_cv_perf 