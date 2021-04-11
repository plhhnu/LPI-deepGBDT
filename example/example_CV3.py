# encoding: utf-8
'''
@author: Bailiqianye
@license: (C) Copyright 2020-2023.
@contact: wangzhaodl@163.com
@time: 2020/12/7
@LPI-deepGBDT: A Multiple-layer Deep Framework based on Gradient Boosting Decision Trees for lncRNA-protein Interaction Identification
'''

import  pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,recall_score,precision_score
from sklearn.metrics import f1_score
from rna_pr_gbdt import RNA_PR_GBDT
from utils.log_utils import logger
from sklearn.model_selection import StratifiedKFold


#load data
Related = pd.read_csv("../example/data/Related_948.csv", header=None).to_numpy()
RNA_feature_935 = pd.read_csv("../example/data/RNA_feature_109.csv", header=None).to_numpy()
Protein_feature_59 = pd.read_csv("../example/data/Protein_feature_35.csv", header=None).to_numpy()


def perform(y_proba,y_true):    #Performance evaluation
     ACC = accuracy_score(y_true, y_proba.argmax(axis=1))
     fpr, tpr, threshold = roc_curve(y_true, y_proba[:, 1])
     pre, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
     RECALL = recall_score(y_true, y_proba.argmax(axis=1))
     AUC = auc(fpr, tpr)
     AUPR = auc(rec, pre)
     f1 = f1_score(y_true, y_proba.argmax(axis=1))
     Pre = precision_score(y_true, y_proba.argmax(axis=1))
     print("ACC=", ACC)
     print('AUC=', AUC)
     print("AUPR=", AUPR)
     print("RECALL=", RECALL)
     print("f1=", f1)
     print("Pre=", Pre)
     return ACC,AUC,AUPR,RECALL,f1,Pre

def creat_data():
    true = np.argwhere(Related == 1)
    temp_t = np.ones(true.shape[0])
    false = np.argwhere(Related == 0)
    temp_f = np.zeros(false.shape[0])
    true_lable = np.insert(true, 2, values=temp_t, axis=1)
    false_lable = np.insert(false, 2, values=temp_f, axis=1)
    np.random.shuffle(false_lable)
    false_lable_temp = false_lable[:true.shape[0]]
    related_index = np.vstack((true_lable, false_lable_temp))
    np.random.shuffle(related_index)
    related_yes_no = related_index[:, 2]
    # related_inde[:,0]--RNA    related_inde[:,1]--Protein
    RNA_index, Protein_index = related_index[:, 0], related_index[:, 1]
    RNA_feature = np.zeros((1, 100))
    Protein_feature = np.zeros((1, 100))
    for i in RNA_index: RNA_feature = np.vstack([RNA_feature, RNA_feature_935[i]])
    RNA_feature = RNA_feature[1:]
    for j in Protein_index: Protein_feature = np.vstack([Protein_feature, Protein_feature_59[j]])
    Protein_feature = Protein_feature[1:]
    feature = np.hstack([Protein_feature, RNA_feature])
    feature = feature.astype(np.float64)
    yes_no = related_yes_no.astype(np.int32)
    return feature,yes_no

if __name__ == "__main__":
    X,y= creat_data()
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    accs = []
    for ci, (train_index, test_index) in enumerate(skf.split(X, y)):
        logger.info("[progress] cv={}/{}".format(ci + 1, 5))
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # y_train = y_train.reshape(len(y_train), 1)
        # y_test = y_test.reshape(len(y_test), 1)
        logger.info("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
        logger.info("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))
        y_proba,y_true = RNA_PR_GBDT(x_train, x_test, y_train, y_test)
        ACC, AUC, AUPR, RECALL, f1, Pre = perform(y_proba, y_true)





