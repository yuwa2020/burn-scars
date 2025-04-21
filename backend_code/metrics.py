import statistics as stats
import numpy as np
from sklearn.metrics import confusion_matrix



class Evaluator():
    """
    The Evaluator currently can do the following metrics:
        - Precision
        - Recall
        - Fscore
    """

    def __init__(self):

        # Declare Metrics
        self.NOT_BURN_SCAR_ACC = 0
        self.BURN_SCAR_ACC = 0
        
        self.NOT_BURN_SCAR_PRECISION = 0
        self.BURN_SCAR_PRECISION = 0
        
        self.NOT_BURN_SCAR_RECALL = 0
        self.BURN_SCAR_RECALL = 0
        
        self.NOT_BURN_SCAR_FSCORE = 0
        self.BURN_SCAR_FSCORE = 0
    
    def run_eval(self, pred_unpadded, gt_labels):
        
        cm = confusion_matrix(gt_labels.flatten(), pred_unpadded.flatten(), labels = [0, 1, -1])
        TP_0 = cm[0][0]
        FP_0 = cm[1][0]
        FN_0 = cm[0][1]
        TN_0 = cm[1][1]
        
        
        TP_1 = cm[1][1]
        FP_1 = cm[0][1]
        FN_1 = cm[1][0]
        TN_1 = cm[0][0]
        
        
        ####DRY
        self.NOT_BURN_SCAR_ACC = ((TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0))*100
        print("NOT_BURN_SCAR Accuracy: ", self.NOT_BURN_SCAR_ACC)
        self.NOT_BURN_SCAR_PRECISION = ((TP_0)/(TP_0+FP_0))*100
        print("NOT_BURN_SCAR Precision: ", self.NOT_BURN_SCAR_PRECISION)
        self.NOT_BURN_SCAR_RECALL = ((TP_0)/(TP_0+FN_0))*100
        print("NOT_BURN_SCAR Recall: ", self.NOT_BURN_SCAR_RECALL)
        self.NOT_BURN_SCAR_FSCORE = ((2*self.NOT_BURN_SCAR_PRECISION*self.NOT_BURN_SCAR_RECALL)/(self.NOT_BURN_SCAR_PRECISION+self.NOT_BURN_SCAR_RECALL))
        print("NOT_BURN_SCAR F-score: ", self.NOT_BURN_SCAR_FSCORE)
        
        print("\n")
        
        ####FLOOD
        self.BURN_SCAR_ACC = ((TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1))*100
        print("BURN_SCAR Accuracy: ", self.BURN_SCAR_ACC)
        self.BURN_SCAR_PRECISION = ((TP_1)/(TP_1+FP_1))*100
        print("BURN_SCAR Precision: ", self.BURN_SCAR_PRECISION)
        self.BURN_SCAR_RECALL = ((TP_1)/(TP_1+FN_1))*100
        print("BURN_SCAR Recall: ", self.BURN_SCAR_RECALL)
        self.BURN_SCAR_FSCORE = ((2*self.BURN_SCAR_PRECISION*self.BURN_SCAR_RECALL)/(self.BURN_SCAR_PRECISION+self.BURN_SCAR_RECALL))
        print("BURN_SCAR F-score: ", self.BURN_SCAR_FSCORE)

        not_burn_scar_acc = float("{:.2f}".format(self.NOT_BURN_SCAR_ACC))
        not_burn_scar_precision = float("{:.2f}".format(self.NOT_BURN_SCAR_PRECISION))
        not_burn_scar_recall = float("{:.2f}".format(self.NOT_BURN_SCAR_RECALL))
        not_burn_scar_f1 = float("{:.2f}".format(self.NOT_BURN_SCAR_FSCORE))
        # dry_iou = float("{:.2f}".format(self.DRY_IOU))

        burn_scar_precision = float("{:.2f}".format(self.BURN_SCAR_PRECISION))
        burn_scar_recall = float("{:.2f}".format(self.BURN_SCAR_RECALL))
        burn_scar_f1 = float("{:.2f}".format(self.BURN_SCAR_FSCORE))
        # flood_iou = float("{:.2f}".format(self.FLOOD_IOU))

        metrices_str = "   Metrics (Unit: %)    "
        metrices_str += "\n\n"
        metrices_str += f"Accuracy         : {not_burn_scar_acc}"
        metrices_str += "\n\n"
        metrices_str += f"NOT_BURN_SCAR Precision    : {not_burn_scar_precision}"
        metrices_str += "\n"
        metrices_str += f"NOT_BURN_SCAR Recall       : {not_burn_scar_recall}"
        metrices_str += "\n"
        metrices_str += f"NOT_BURN_SCAR F1 score     : {not_burn_scar_f1}"
        # metrices_str += "\n"
        # metrices_str += f"Dry IOU          : {dry_iou}"
        metrices_str += "\n\n"
        metrices_str += f"BURN_SCAR Precision  : {burn_scar_precision}"
        metrices_str += "\n"
        metrices_str += f"BURN_SCAR Recall     : {burn_scar_recall}"
        metrices_str += "\n"
        metrices_str += f"BURN_SCAR F1 score   : {burn_scar_f1}"

        return metrices_str

        
    
    
    @property
    def f_accuracy(self):        
        if self.BURN_SCAR_ACC > 0:
            return self.BURN_SCAR_ACC
        else:
            return 0.0

    @property
    def f_precision(self):        
        if self.BURN_SCAR_PRECISION > 0:
            return self.BURN_SCAR_PRECISION
        else:
            return 0.0

 
    @property
    def f_recall(self):
        if self.BURN_SCAR_RECALL > 0:
            return self.BURN_SCAR_RECALL
        else:
            return 0.0
        
        
    @property
    def f_fscore(self):
        if self.BURN_SCAR_FSCORE > 0:
            return self.BURN_SCAR_FSCORE
        else:
            return 0.0
    
    
    
    
    @property
    def d_accuracy(self):        
        if self.NOT_BURN_SCAR_ACC > 0:
            return self.NOT_BURN_SCAR_ACC
        else:
            return 0.0
    
    @property
    def d_precision(self):        
        if self.NOT_BURN_SCAR_PRECISION > 0:
            return self.NOT_BURN_SCAR_PRECISION
        else:
            return 0.0

 
    @property
    def d_recall(self):
        if self.NOT_BURN_SCAR_RECALL > 0:
            return self.NOT_BURN_SCAR_RECALL
        else:
            return 0.0
        
        
    @property
    def d_fscore(self):
        if self.NOT_BURN_SCAR_FSCORE > 0:
            return self.NOT_BURN_SCAR_FSCORE
        else:
            return 0.0
