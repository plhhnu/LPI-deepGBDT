# encoding: utf-8
'''
@author: Bailiqianye
@license: (C) Copyright 2020-2023.
@contact: wangzhaodl@163.com
@time: 2020/12/7
@LPI-deepGBDT: A Multiple-layer Deep Framework based on Gradient Boosting Decision Trees for lncRNA-protein Interaction Identification
'''
from pandas import DataFrame


def put_csv(matrix):
    # 输出到CSV文件
    ss = DataFrame(matrix)
    ss.to_csv("temp.csv", index=False, header=None, sep=',')
