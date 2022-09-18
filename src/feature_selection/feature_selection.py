import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif
)
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from ReliefF import ReliefF
from src.cross_validation.cross_validation import kfold
from src.performance.performance import PerformanceMetrics

class FeatureSelection():

    def __init__(self):
        return

    def variance_threshold(self, clf, X: pd.DataFrame, y: list, thr: float = 0, baseline: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().variance_threshold(clf, X, y, thr=0.5, baseline=True)
        """

        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)

        # 2. Apply feature selection method
        X_sel = pd.DataFrame(VarianceThreshold(threshold=thr).fit_transform(X))

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X_sel, y, k=5, name='variance_threshold')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, X_sel.columns)

        return fs_confusion_matrix, performance, df_feature_selected

    def anova(self, clf, X: pd.DataFrame, y: list, n_feat: int, baseline: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().anova(clf, X, y, n_feat=30, baseline=True)
        """
        
        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)

        # 2. Apply feature selection method
        fs = SelectKBest(score_func=f_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X_sel, y, k=5, name='anova')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, X_sel.columns)

        return fs_confusion_matrix, performance, df_feature_selected

    def mutual_info(self, clf, X: pd.DataFrame, y: list, n_feat: int, baseline: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().mutual_info(clf, X, y, n_feat=30, baseline=True)
        """
        
        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)

        # 2. Apply feature selection method
        fs = SelectKBest(score_func=mutual_info_classif, k=n_feat)
        X_sel = pd.DataFrame(fs.fit_transform(X, y))

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X_sel, y, k=5, name='mutual_information')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, X_sel.columns)

        return fs_confusion_matrix, performance, df_feature_selected

    def recursive_feature_elimination(self, clf, X: pd.DataFrame, y: list, n_feat: int, baseline: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().recursive_feature_elimination(clf, X, y, n_feat=30, baseline=True)
        """
        
        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)

        # 2. Apply feature selection method
        rfe_selector = RFE(estimator=clf, n_features_to_select=n_feat, step=1)
        rfe_selector.fit(X, y)
        feat_to_select = X.columns[rfe_selector.get_support()]

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X[feat_to_select], y, k=5, name='rfe')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, feat_to_select)

        return fs_confusion_matrix, performance, df_feature_selected

    def random_forest_importance(self, clf, X: pd.DataFrame, y: list, threshold: float = 0.95, baseline: bool = False, verbose: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().random_forest_importance(clf, X, y, threshold=0.8, baseline=True, verbose=True)
        """

        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)

        # 2. Apply feature selection method
        rf = RandomForestClassifier()
        rf.fit(X.values, y.values)
        df_feature_importance = pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": rf.feature_importances_
            }
        ).sort_values("importance", ascending=False)

        df_feature_importance["cumsum_importance"] = np.cumsum(df_feature_importance["importance"])
        feat_to_select = np.array(df_feature_importance[np.cumsum(df_feature_importance["importance"]) <= threshold]["feature"])
        
        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X[feat_to_select], y, k=5, name='rf_importance')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, feat_to_select)

        if (verbose):
            plt.figure()
            sns.barplot(x=df_feature_importance.feature, y=df_feature_importance.importance)
            sns.lineplot(x=df_feature_importance.feature, y=df_feature_importance.cumsum_importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance Score")
            plt.title("Random Forest Importance")
            plt.grid(b=True)
            plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
            plt.show(block=False)
        
        return fs_confusion_matrix, performance, df_feature_selected

    def relieff(self, clf, X: pd.DataFrame, y: list, n_feat: int, baseline: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().relieff(clf, X, y, n_feat=30, baseline=True)
        """

        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)
        
        # 2. Apply feature selection method
        fs = ReliefF(n_neighbors=20, n_features_to_keep=n_feat)
        X_sel = fs.fit_transform(X.values, y.values)

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X_sel, y, k=5, name='relieff')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, X_sel.columns)

        return fs_confusion_matrix, performance, df_feature_selected

    def correlation_removal():
        #TODO:implement the method.
        pass

    def cluster_quality(self, clf, X: pd.DataFrame, y: list, n_feat: int, baseline: bool = False, verbose: bool = False):
        """
        conf_matrix, fs_perf, feat_selected = FeatureSelection().cluster_quality(clf, X, y, n_feat=30, baseline=True, verbose=True)
        """
        performance = []

        if baseline:
            # 1. Define the baseline perfornace using the input clf and all the feature set
            baseline_confusion_matrix, baseline_perf = kfold(clf, X, y, k=5, cv_name='baseline')
            performance.append(baseline_perf)
        
        # 2. Apply feature selection method
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

        feat_to_select = df_feature_importance['feature'][0:n_feat]

        if (verbose):
            plt.figure()
            sns.barplot(x=df_feature_importance.feature, y=df_feature_importance.importance)
            plt.xlabel("Features")
            plt.ylabel("Feature Importance Score")
            plt.title("Davies-Bouldin importance")
            plt.grid(b=True)
            plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="x-large")
            plt.show(block=False)

        # 3. Define the feature selection performance
        fs_confusion_matrix, fs_perf = kfold(clf, X[feat_to_select], y, k=5, name='cluster_quality')
        performance.append(fs_perf)

        df_feature_selected = self.selected_features_table(X.columns, feat_to_select)

        return fs_confusion_matrix, performance, df_feature_selected

    def selected_features_table(self, feature_list, selected_features):

        is_selected = [(x in list(selected_features)) for x in list(feature_list)]

        df_feature = pd.DataFrame(
            {
                "feature": feature_list,
                "is_selected": is_selected
            }
        )

        return df_feature