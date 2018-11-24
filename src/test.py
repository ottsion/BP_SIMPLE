#!/usr/bin/env python
# -*- coding:utf-8 -*-

#
# author: sunfc
# Create At: 2018-11-22
#
from src import bp_model
import numpy as np


if __name__ == "__main__":
    datas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([
        [1.0],
        [0.0],
        [0.0],
        [1.0]
    ])
    bp = bp_model.bpModel([2, 4, 1], 6000)
    bp.train(datas, labels)
    print(bp.predict([3, 1]))
