import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import davies_bouldin_score

## Feature Selection Classes
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF

from src.cross_validation.cross_validation import CrossValidation
from src.performance.performance import PerformanceMetrics

class FeatureSelection():

    def __init__(self):
        return

    def variance_threshold(self, clf, X, y, thr=0):
        # Variance Threshold Feature Selection
        X_sel = pd.DataFrame(VarianceThreshold(threshold=thr).fit_transform(X))

        # Davies-Bouldin Score (after MUTUAL INFORMATION feature selection)
        db_score = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        print("\nVARIANCE THR - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X_sel, y).kfold(clf, k=10, cv_name='var_thr')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf

    def anova(self, clf, X, y, n_feat):
        # ANOVA Feature Selection
        fs = SelectKBest(score_func=f_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # Davies-Bouldin Score (after ANOVA feature selection)
        db_score = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        print("\nANOVA - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X_sel, y).kfold(clf, k=10, cv_name='anova')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf

    def mutual_info(self, clf, X, y, n_feat):
        # MUTUAL INFORMATION Feature Selection
        fs = SelectKBest(score_func=mutual_info_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # Davies-Bouldin Score (after MUTUAL INFORMATION feature selection)
        db_score = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        print("\nMUTUAL INFO - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X_sel, y).kfold(clf, k=10, cv_name='mutual_info')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf

    def rfe(self, clf, X, y):
        # RFE Feature Selection
        rfe_selector = RFE(estimator = clf, step = 1)
        rfe_selector.fit(X, y)
        feat_to_select = X.columns[rfe_selector.get_support()]
        print(feat_to_select)

        db_score = davies_bouldin_score(StandardScaler().fit(X[feat_to_select]).transform(X[feat_to_select]), y)

        # 10-fold cross-validation
        print("\nRFE - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X[feat_to_select], y).kfold(clf, k=10, cv_name='rfe')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf

    def rf_feature_importance(self, clf, X, y, threshold=0.95, verbose=False):
        # Random Forest Feature Importance Feature Selection
        clf.fit(X.values, y.values)
        df_feature_importance = pd.DataFrame({"feature": list(X.columns), "importance": clf.feature_importances_}
            ).sort_values("importance", ascending=False)

        df_feature_importance["cumsum_importance"] = np.cumsum(df_feature_importance["importance"])

        selected_features = np.array(df_feature_importance[np.cumsum(df_feature_importance["importance"]) <= threshold]["feature"])

        db_score = davies_bouldin_score(StandardScaler().fit(X[selected_features]).transform(X[selected_features]), y)

        print("\nFEAT IMPORTANCE - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X[selected_features], y).kfold(clf, k=10, cv_name='rf_feature_importance')
        df_perf['cluster_quality'] = db_score

        if(verbose):
            plt.figure()
            sns.barplot(x=df_feature_importance.feature, y=df_feature_importance.importance)
            sns.lineplot(x=df_feature_importance.feature, y=df_feature_importance.cumsum_importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance Score")
            plt.title("Random Forest Feature Importance")
            plt.grid(b = True)
            plt.xticks(rotation = 45, horizontalalignment = "right", fontweight = "light", fontsize = "x-large")
            plt.show(block = False)
        
        return kfold_confusion_matrix, df_perf, selected_features

    def relieff(self, clf, X, y, n_feat):
        # ReliefF Feature Selection
        fs = ReliefF(n_neighbors=20, n_features_to_keep=n_feat)
        X_sel = fs.fit_transform(X, y)

        db_score = davies_bouldin_score(StandardScaler().fit(X[X_sel]).transform(X[X_sel]), y)

        print("\nRELIEFF - 10-fold-cross-validation...\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X_sel, y).kfold(clf, k=10, cv_name='relieff')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf

    def correlation_removal():

        pass

    def cluster_quality(self, clf, X, y, n_feat):
        # Cluster Quality Feature Selection
        X_scaled = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns)

        db_importance = np.zeros(len(X.columns))
        cnt = 0
        for feature in X.columns:
            db_importance[cnt] = davies_bouldin_score(np.array(X_scaled[feature]).reshape(-1,1), y)
            cnt = cnt + 1

        df_feature_importance = pd.DataFrame({"feature": list(X.columns), "importance": db_importance}
            ).sort_values("importance", ascending=True)

        selected_features = df_feature_importance['feature'][0:n_feat]

        # plt.figure()
        # sns.barplot(x=feature_importances_db.feature, y=feature_importances_db.importance)
        # plt.xlabel("Features")
        # plt.ylabel("Feature Importance Score")
        # plt.title("Davies-Boulding feature importance")
        # plt.grid(b = True)
        # plt.xticks(rotation = 45, horizontalalignment = "right", fontweight = "light", fontsize = "x-large")
        # plt.show(block = False)

        db_score = davies_bouldin_score(StandardScaler().fit(X[selected_features]).transform(X[selected_features]), y)

        print("\nCLUSTER QUALITY - 10-fold-cross-validation\n")
        kfold_confusion_matrix, df_perf = CrossValidation(X[selected_features], y).kfold(clf, k=10, cv_name='davies_bouldin')
        df_perf['cluster_quality'] = db_score

        return kfold_confusion_matrix, df_perf
