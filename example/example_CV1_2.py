# encoding: utf-8
'''
@author: Bailiqianye
@license: (C) Copyright 2020-2023.
@contact: wangzhaodl@163.com
@time: 2020/12/7
@LPI-deepGBDT: A Multiple-layer Deep Framework based on Gradient Boosting Decision Trees for lncRNA-protein Interaction Identification
'''
from collections import defaultdict
import warnings
import  pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,recall_score,precision_score
from sklearn.metrics import f1_score
from deepGBDT import RNA_PR_GBDT
import numpy as np
warnings.filterwarnings("ignore")
def Cut_array(c, d):
    for i in range(len(c)):
        for j in c:
            if j in d:
                d.remove(j)
    return d
def balance(data):
    index_y = np.zeros((1, 2))
    index_n = np.zeros((1, 2))
    x, y = data[:, 1], data[:, 0]
    test_label = Related[x, y]
    num = np.sum(test_label)
    for i in range(len(test_label)):
        if (test_label[i] == 1):
            index_y = np.vstack([index_y, data[i]])
        else:
            index_n = np.vstack([index_n, data[i]])
    index_y, index_n = index_y[1:], index_n[1:]
    np.random.shuffle(index_n)
    index_n = index_n[:int(num)]
    index = np.vstack([index_y, index_n]).astype(np.int32)
    return index


def lable(index):
    x, y = index[:, 1], index[:, 0]
    test_label = Related[x, y]
    return np.array(test_label)


def generate_feature(index):
    RNA_feature_935 = pd.read_csv("./example/data/RNA_feature_109.csv", header=None).to_numpy()
    Protein_feature_59 = pd.read_csv("./example/data/Protein_feature_35.csv", header=None).to_numpy()
    RNA_feature = np.zeros((1, 100))
    Protein_feature = np.zeros((1, 100))
    RNA_index, Protein_index = index[:, 1], index[:, 0]
    for i in RNA_index: RNA_feature = np.vstack([RNA_feature, RNA_feature_935[i]])
    RNA_feature = RNA_feature[1:]
    for j in Protein_index: Protein_feature = np.vstack([Protein_feature, Protein_feature_59[j]])
    Protein_feature = Protein_feature[1:]
    feature = np.hstack([Protein_feature, RNA_feature]).astype(np.float64)
    print("generate-feature-completed")
    return feature


def cross_validation(intMat, seeds, cv, num=5):  # 交叉验证
    cv_data = defaultdict(list)
    for seed in seeds:
        num_row, num_col = intMat.shape
        print(num_row, num_col)
        prng = np.random.RandomState(seed)
        if cv == 1:
            index = prng.permutation(num_row)
        if cv == 2:
            index = prng.permutation(num_col)
        step = index.size // num
        for i in range(num):
            if i < num - 1:
                ii = index[i * step:(i + 1) * step]
            else:
                ii = index[i * step:]
            if cv == 1:
                print(ii)
                train_ii = np.array(Cut_array(list(ii), list(index)))
                print(train_ii)
                test_data = np.array([[k, j] for k in ii for j in range(num_col)], dtype=np.int32)
                train_data = np.array([[k, j] for k in train_ii for j in range(num_col)], dtype=np.int32)
                test_data, train_data = test_data[:, [1, 0]], train_data[:, [1, 0]]
            elif cv == 2:
                print(ii)
                train_ii = np.array(Cut_array(list(ii), list(index)))
                print(train_ii)
                test_data = np.array([[k, j] for k in ii for j in range(num_row)], dtype=np.int32)
                train_data = np.array([[k, j] for k in train_ii for j in range(num_row)], dtype=np.int32)
            ####################################################################################################
            test_index = balance(test_data)
            train_index = balance(train_data)
            test_label = lable(test_index).astype(np.int32)
            train_label = lable(train_index).astype(np.int32)
            print("Test-Positive and negative sample balance:", len(test_label) - np.sum(test_label) * 2)
            print("Train-Positive and negative sample balance:", len(train_label) - np.sum(train_label) * 2)
            cv_data[seed].append((train_index, train_label, test_index, test_label))
    return cv_data


def perform(y_proba,y_true):    #Performance evaluation
     ACC = accuracy_score(y_true, y_proba.argmax(axis=1))
     fpr, tpr, threshold = roc_curve(y_true, y_proba[:, 1])
     pre, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
     RECALL = recall_score(y_true, y_proba.argmax(axis=1))
     AUC = auc(fpr, tpr)
     AUPR = auc(rec, pre)
     f1 = f1_score(y_true, y_proba.argmax(axis=1))
     Pre = precision_score(y_true, y_proba.argmax(axis=1))
     return ACC,AUC,AUPR,RECALL,f1,Pre

def train(data):
    ACC_array = []
    AUC_array = []
    RECALL_array = []
    AUPR_array = []
    f1_array = []
    Pre_array = []
    for seed in data.keys():
        for train_index,train_label,test_index,test_label in data[seed]:
            X_train = generate_feature(train_index)
            X_test = generate_feature(test_index)
            y_proba, y_true = RNA_PR_GBDT(X_train, X_test, train_label, test_label)
            ACC,AUC,AUPR,RECALL,f1,Pre = perform(y_proba, y_true)
            ACC_array.append(ACC)
            AUC_array.append(AUC)
            AUPR_array.append(AUPR)
            RECALL_array.append(RECALL)
            f1_array.append(f1)
            Pre_array.append(Pre)
    print("#########%%%%%%%Five-fold cross validation—Result%%%%%%%%%%%#####")
    print("ACC=", ACC_array)
    print('AUC=', AUC_array)
    print("AUPR=", AUPR_array)
    print("RECALL=", RECALL_array)
    print("f1=", f1_array)
    print("Pre=", Pre_array)
    print("#######################%%%%%%%%%%%%%%%%%%#######################")
    return ACC_array,AUC_array,AUPR_array,RECALL_array,f1_array,Pre_array


if __name__ == "__main__":
    # ACC_array_20 = []
    # AUC_array_20 = []
    # RECALL_array_20 = []
    # AUPR_array_20 = []
    # f1_array_20 = []
    # Pre_array_20 = []
    #for i in range(20):
    seeds = [np.random.randint(10000)]
    Related = pd.read_csv("./example/data/Related_948.csv", header=None).to_numpy()
    cv_data = cross_validation(Related, seeds, cv=2, num=5)
    ACC,AUC,AUPR,RECALL,f1,Pre = train(cv_data)
    #     ACC_array_20.append(np.mean(ACC))
    #     AUC_array_20.append(np.mean(AUC))
    #     AUPR_array_20.append(np.mean(AUPR))
    #     RECALL_array_20.append(np.mean(RECALL))
    #     f1_array_20.append(np.mean(f1))
    #     Pre_array_20.append(np.mean(Pre))
    # ACC_sum_20 = np.mean(ACC_array_20)
    # AUC_sum_20 = np.mean(AUC_array_20)
    # AUPR_sum_20 = np.mean(AUPR_array_20)
    # RECALL_sum_20 = np.mean(RECALL_array_20)
    # f1_sum_20 =np.mean(f1_array_20)
    # Pre_sum_20 = np.mean(Pre_array_20)
    # print(ACC_array_20)
    # print(AUC_array_20)
    # print(AUPR_array_20)
    # print(RECALL_array_20)
    # print(f1_array_20)
    # print(Pre_array_20)
    # print("###############################20-Five-fold cross validation—Result############################################")
    # print("20-Five-fold cross validation—Result--ACC:", '%.4f' % ACC_sum_20, "±", '%.4f' % np.std(ACC_array_20))
    # print("20-Five-fold cross validation—Result--AUC:", '%.4f' % AUC_sum_20, "±", '%.4f' % np.std(AUC_array_20))
    # print("20-Five-fold cross validation—Result--AUPR:", '%.4f' % AUPR_sum_20, "±", '%.4f' % np.std(AUPR_array_20))
    # print("20-Five-fold cross validation—Result--RECALL:", '%.4f' % RECALL_sum_20, "±",'%.4f' % np.std(RECALL_array_20))
    # print("20-Five-fold cross validation—Result--f1：", '%.4f' % f1_sum_20, "±", '%.4f' % np.std(f1_array_20))
    # print("20-Five-fold cross validation—Result--Pre", '%.4f' % Pre_sum_20, "±", '%.4f' % np.std(Pre_array_20))






