import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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

from src.cross_validation.cross_validation import kfold
from src.performance.performance import PerformanceMetrics

class FeatureSelection():

    def __init__(self):
        return

    def variance_threshold(self, clf, X, y, thr: float = 0):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)
        
        # Variance Threshold Feature Selection
        X_sel = pd.DataFrame(VarianceThreshold(threshold=thr).fit_transform(X))

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X_sel, y, k=10, cv_name='var_thr')

        return kfold_confusion_matrix, df_perf, db_score_after, db_score_before

    def anova(self, clf, X, y, n_feat: int):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)

        # ANOVA Feature Selection
        fs = SelectKBest(score_func=f_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X_sel, y, k=10, cv_name='anova')

        return kfold_confusion_matrix, df_perf, db_score_before, db_score_after

    def mutual_info(self, clf, X, y, n_feat: int):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)

        # MUTUAL INFORMATION Feature Selection
        fs = SelectKBest(score_func=mutual_info_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X_sel, y, k=10, cv_name='mutual_info')

        return kfold_confusion_matrix, df_perf, db_score_before, db_score_after

    def recursive_feature_elimination(self, clf, X, y):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)
        
        # RFE Feature Selection
        rfe_selector = RFE(estimator=clf, step=1)
        rfe_selector.fit(X, y)
        feat_to_select = X.columns[rfe_selector.get_support()]

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X[feat_to_select]).transform(X[feat_to_select]), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X[feat_to_select], y, k=10, cv_name='rfe')

        return feat_to_select, kfold_confusion_matrix, df_perf, db_score_before, db_score_after 

    def random_forest_feature_importance(self, clf, X, y, threshold: float = 0.95, verbose: bool = False):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)
        
        # Random Forest Feature Importance Feature Selection

        model = RandomForestClassifier()
        model.fit(X.values, y.values)
        df_feature_importance = pd.DataFrame(
            {
                "feature":list(X.columns),
                "importance":model.feature_importances_
            }
        ).sort_values("importance", ascending=False)

        df_feature_importance["cumsum_importance"] = np.cumsum(df_feature_importance["importance"])

        selected_features = np.array(df_feature_importance[np.cumsum(df_feature_importance["importance"]) <= threshold]["feature"])

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X[selected_features]).transform(X[selected_features]), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X[selected_features], y, k=10, cv_name='rf_feature_importance')

        if (verbose):
            plt.figure()
            sns.barplot(x=df_feature_importance.feature, y=df_feature_importance.importance)
            sns.lineplot(x=df_feature_importance.feature, y=df_feature_importance.cumsum_importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance Score")
            plt.title("Random Forest Feature Importance")
            plt.grid(b=True)
            plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
            plt.show(block=False)
        
        return selected_features, kfold_confusion_matrix, df_perf, db_score_before, db_score_after

    def relieff(self, clf, X, y, n_feat: int):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)
        
        # ReliefF Feature Selection
        fs = ReliefF(n_neighbors=20, n_features_to_keep=n_feat)
        X_sel = fs.fit_transform(X, y)

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X_sel).transform(X_sel), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X_sel, y, k=10, cv_name='ReliefF')

        return kfold_confusion_matrix, df_perf, db_score_before, db_score_after

    def correlation_removal():
        pass

    def cluster_quality(self, clf, X, y, n_feat: int, verbose: bool = False):
        
        # Davies-Bouldin Score (before the feature selection method)
        db_score_before = davies_bouldin_score(StandardScaler().fit(X).transform(X), y)
        
        # Cluster Quality Feature Selection
        X_scaled = pd.DataFrame(StandardScaler().fit(X).transform(X), columns = X.columns)

        db_importance = np.zeros(len(X.columns))
        cnt = 0
        for feature in X.columns:
            db_importance[cnt] = davies_bouldin_score(np.array(X_scaled[feature]).reshape(-1,1), y)
            cnt = cnt + 1

        df_feature_importance = pd.DataFrame(
            {
                "feature":list(X.columns),
                "importance":db_importance
            }
        ).sort_values("importance", ascending=True)

        selected_features = df_feature_importance['feature'][0:n_feat]

        if (verbose):
            plt.figure()
            sns.barplot(x=df_feature_importance.feature, y=df_feature_importance.importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance Score")
            plt.title("Davies-Boulding feature importance")
            plt.grid(b=True)
            plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
            plt.show(block=False)

        # Davies-Bouldin Score (after the feature selection method)
        db_score_after = davies_bouldin_score(StandardScaler().fit(X[selected_features]).transform(X[selected_features]), y)

        # 10-fold cross-validation
        kfold_confusion_matrix, df_perf = kfold(clf, X[selected_features], y, k=10, cv_name='davies_bouldin')

        return selected_features, kfold_confusion_matrix, df_perf, db_score_before, db_score_after
