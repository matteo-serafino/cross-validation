import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay

class PerformanceMetrics():

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = y_true.unique()
        self.n_labels = len(self.labels)
        self.n_labels_pred = len(np.unique(y_pred))

        return

    def classification_scores(self, verbose: bool = False):

        # Binary classification use case        
        if (max(self.n_labels, self.n_labels_pred) == 2):

            conf_mat = confusion_matrix(self.y_true, self.y_pred)
            

            result = pd.DataFrame.from_dict({
                "precision": np.round(conf_mat[0,0] / (conf_mat[0,0] + conf_mat[1,0]), 2),
                "sensitivity": np.round(conf_mat[0,0] / (conf_mat[0,0] + conf_mat[0,1]), 2),
                "specificity": np.round(conf_mat[1,1] / (conf_mat[1,1] + conf_mat[1,0]), 2),
                "f1_score": np.round(f1_score(self.y_true, self.y_pred, average= 'micro'), 2)})
            

            clf_report_dict = classification_report(self.y_true, self.y_pred, output_dict=True)
            

            if verbose:
                print(f"Confusion Matrix:\n{conf_mat}")
                print(f"Classification metrics:\n{result}")
                print(f"Classification report:\n{classification_report(self.y_true, self.y_pred)}")
            
            ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred)
            plt.show(block=False)

        # Multi-class classification use case
        elif (max(self.n_labels, self.n_labels_pred) >= 3):
           
            conf_mat = confusion_matrix(self.y_true, self.y_pred)

            # precision = conf_mat[0,0] / (conf_mat[0,0] + conf_mat[1,0] + conf_mat[2,0])
            # sensitivity = conf_mat[0,0] / (conf_mat[0,0] + conf_mat[0,1] + conf_mat[0,2])
            # specificity_N = conf_mat[1,1] / (conf_mat[1,1] + conf_mat[1,0] + conf_mat[1, 2])
            # specificity_O = conf_mat[2,2] / (conf_mat[2,2] + conf_mat[2,0] + conf_mat[2, 1])

            # result = {
            #     "sensitivity": np.round(sensitivity, 2),
            #     "precision": np.round(precision, 2), 
            #     "specificity_n": np.round(specificity_N, 2),
            #     "specificity_o": np.round(specificity_O, 2),
            #     "sklearn_f1_score": np.round(f1_score(self.y_true, self.y_pred, average= 'macro'), 2)
            # }

            # print("Classification metrics:")
            # print(result)

            clf_report_dict = classification_report(self.y_true, self.y_pred, output_dict=True)
            
            if verbose:
                print(f"Confusion Matrix:\n{conf_mat}")
                print(f"Classification report:\n{classification_report(self.y_true, self.y_pred)}")

            ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred)
            plt.show(block=False)
        else:
            pass
    
        return conf_mat, clf_report_dict

    def confusion_matrix_plot(self):

        ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred)
        plt.show(block=False)

        return