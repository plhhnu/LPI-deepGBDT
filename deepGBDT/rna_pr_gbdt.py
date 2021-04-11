# encoding: utf-8
'''
@author: Bailiqianye
@license: (C) Copyright 2020-2023.
@contact: wangzhaodl@163.com
@time: 2020/12/7
@LPI-deepGBDT: A Multiple-layer Deep Framework based on Gradient Boosting Decision Trees for lncRNA-protein Interaction Identification
'''
import sys
import torch
from deepGBDT import MGBDT
from deepGBDT import MultiXGBModel, LinearModel
from deepGBDT.utils.log_utils import logger
import matplotlib.pyplot as  plt
from sklearn.model_selection import StratifiedKFold
from deepGBDT.utils.exp_utils import set_seed

def RNA_PR_GBDT(x_train, x_test, y_train, y_test):
 set_seed(None)
 n_layers = 3
 logger.info("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
 logger.info("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))
 net = MGBDT(loss=None, target_lr=1.0, epsilon=0.3)
 #####################################################################################
 net.add_layer("tp_layer",
               F=MultiXGBModel(input_size=200, output_size=16, learning_rate=0.1, max_depth=5, num_boost_round=5),
               G=None)
 for i in range(n_layers - 2):
    net.add_layer("tp_layer",
               F=MultiXGBModel(input_size=16, output_size=16, learning_rate=0.1, max_depth=5,
                               num_boost_round=5),
               G=MultiXGBModel(input_size=16, output_size=16, learning_rate=0.1, max_depth=5,
                               num_boost_round=5))
 net.add_layer("bp_layer", F=LinearModel(input_size=16, output_size=2, learning_rate=0.1, loss="CrossEntropyLoss"))
 logger.info("[net architecture]")
 logger.info(net)
 net.init(x_train, n_rounds=3)
 net.fit(x_train, y_train, n_epochs=15, eval_sets=[(x_test, y_test)], eval_metric="accuracy")
 y_proba = net.forward(x_test)
 y_proba = torch.tensor(y_proba)
 y_proba_end = torch.sigmoid(y_proba)
 y_true = y_test
 return y_proba_end,y_true