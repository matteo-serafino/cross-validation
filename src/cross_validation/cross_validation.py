import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from src.performance.performance import PerformanceMetrics

def kfold(model, X: pd.DataFrame, y: list, k: int = 10, cv_name: str ='k-fold', verbose: bool = False):
    
    k_fold = StratifiedKFold(n_splits = k)
    y_pred = np.array(X.shape[0] * [y[0]])
    ith_fold = 1

    kfold_perf = []

    for train_index, test_index in k_fold.split(X, y):

        # Defining training set and test set for the i-th fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Model training
        model.fit(X_train.values, y_train)
        y_pred[test_index] = model.predict(X_test.values)

        # i-th fold performance
        ith_cv = {
            'name': cv_name,
            'fold': ith_fold,
            'balanced_accuracy': np.round(balanced_accuracy_score(y_test, y_pred[test_index]), decimals=2),
            'f1_score': np.round(f1_score(y_test, y_pred[test_index], average= 'micro'), decimals=2),
            'mcc': np.round(matthews_corrcoef(y_test, y_pred[test_index]), decimals=2)
        }

        kfold_perf.append(ith_cv)
 
        if (verbose):
            print(f"k = {ith_fold}\nbalance acc = {ith_cv['balanced_accuracy']}, f1-score = {ith_cv['f1_score']}, MCC = {ith_cv['mcc']}")
        
        ith_fold += 1
    
    # Overall performances
    overall_cv = {
        'name': cv_name,
        'fold': "overall",
        'balanced_accuracy': np.round(balanced_accuracy_score(y, y_pred), decimals=2),
        'f1_score': np.round(f1_score(y, y_pred, average= 'micro'), decimals=2),
        'mcc': np.round(matthews_corrcoef(y, y_pred), decimals=2)
    }

    kfold_perf.append(overall_cv)

    if (verbose):
        print(f"OVERALL\nbalanced acc = {overall_cv['balanced_accuracy']}, f1-score = {overall_cv['f1_score']}, MCC = {overall_cv['mcc']}")

    conf_mat = confusion_matrix(y, y_pred)

    df_perf = pd.DataFrame(kfold_perf)

    return conf_mat, df_perf

def leave_one_out(model, X: pd.DataFrame, y: list, cv_name: str ='LOOCV', verbose: bool = False):
    
    loo = LeaveOneOut()
    
    y_pred = np.array(X.shape[0] * [y[0]])

    cnt = 1

    loo_perf = []

    for train_index, test_index in loo.split(X):

        # Defining training set and test set for the i-th leave one out
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Model training
        model.fit(X_train.values, y_train)
        y_pred[test_index] = model.predict(X_test.values)

        cnt = cnt + 1
    
    # Overall performances
    overall_cv = {
        'name': cv_name,
        'fold': "overall",
        'balanced_accuracy': np.round(balanced_accuracy_score(y, y_pred), decimals=2),
        'f1_score': np.round(f1_score(y, y_pred, average= 'micro'), decimals=2),
        'mcc': np.round(matthews_corrcoef(y, y_pred), decimals=2)
    }

    if (verbose):
        print(f"OVERALL\nbalanced acc = {overall_cv['balanced_accuracy']}, f1-score = {overall_cv['f1_score']}, MCC = {overall_cv['mcc']}")

    loo_perf.append(overall_cv)

    conf_mat = confusion_matrix(y, y_pred)

    df_perf = pd.DataFrame(loo_perf)

    return conf_mat, df_perf

def leave_one_subject_out(model, X: pd.DataFrame, y: list, subject_ids: list, cv_name: str ='LOSOCV', verbose: bool = False):

    y_pred = np.array(X.shape[0] * [y[0]])

    subject_id = np.unique(subject_ids)

    loso_perf = []

    for s in subject_id:

        test_index = subject_ids == s
        train_index = not(test_index)

        # Defining training set and test set for the i-th fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Model training
        model.fit(X_train.values, y_train)
        y_pred[test_index] = model.predict(X_test.values)

        ith_cv = {
            'name': cv_name,
            'subject': s,
            'balanced_accuracy': np.round(balanced_accuracy_score(y_test, y_pred[test_index]), decimals=2),
            'f1_score': np.round(f1_score(y_test, y_pred[test_index], average= 'micro'), decimals=2),
            'mcc': np.round(matthews_corrcoef(y_test, y_pred[test_index]), decimals=2)
        }

        loso_perf.append(ith_cv)
 
        if (verbose):
            print(f"subject = {s}\nbalance acc = {ith_cv['balanced_accuracy']}, f1-score = {ith_cv['f1_score']}, MCC = {ith_cv['mcc']}")

    # Overall performances
    overall_cv = {
        'name': cv_name,
        'subject': "overall",
        'balanced_accuracy': np.round(balanced_accuracy_score(y, y_pred), decimals=2),
        'f1_score': np.round(f1_score(y, y_pred, average= 'micro'), decimals=2),
        'mcc': np.round(matthews_corrcoef(y, y_pred), decimals=2)
    }

    if (verbose):
        print(f"OVERALL\nbalanced acc = {overall_cv['balanced_accuracy']}, f1-score = {overall_cv['f1_score']}, MCC = {overall_cv['mcc']}")

    loso_perf.append(overall_cv)

    conf_mat = confusion_matrix(y, y_pred)

    df_perf = pd.DataFrame(loso_perf)

    return conf_mat, df_perf