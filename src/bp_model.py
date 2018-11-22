#!/usr/bin/env python
# -*- coding:utf-8 -*-

#
# author: sunfc
# Create At: 2018-11-22

# 这是个简易bp，隐藏层只有一层
#
import numpy as np

class bpModel():

    """
    bp神经网络需要的参数如下：
        @W ： 输入层到隐藏层的权值
        @alpha ： 输入层到隐藏层的截距
        @V ： 隐藏层到输出层的权值
        @beta ： 输入层到隐藏层的截距
        @X ： 输入节点
        @H ： 隐藏层节点
        @Y ： 输出层节点
    """
    def __init__(self, layers_num, epoches):
        assert len(layers_num) == 3, "初始化参数必须为三个"
        self.epoches = epoches
        self.input_size = layers_num[0]
        self.hidden_size = layers_num[1]
        self.output_size = layers_num[2]
        # 初始化权值
        self.X = np.zeros([1, self.input_size])
        self.W = np.random.random_sample(size=(self.input_size, self.hidden_size))
        self.V = np.random.random_sample(size=(self.hidden_size, self.output_size))
        self.H = np.zeros([1, self.hidden_size])
        self.Y = np.zeros([1, self.output_size])

    def train(self, datas, labels):
        for epoch in range(self.epoches):
            for index in datas.shape[0]:
                self.X = datas[index]
                self.Y = labels[index]
                _ = self.forward()
                self.backword()

    def forward(self):
        return _

    def backword(self):
        pass

    def sigmod(self, x):
        return 1.0 / (1.0 - np.exp(-x))

    def predict(self, data):
        self.X = data
        return self.forward()

