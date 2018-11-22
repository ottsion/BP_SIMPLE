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
        self.H = np.zeros(self.hidden_size)
        self.Y = np.zeros(self.output_size)
        self.alpha = np.random.random
        self.beta = np.random.random
        self.label = []

    def train(self, datas, labels):
        for epoch in range(self.epoches):
            error = 0
            for index in range(datas.shape[0]):
                self.X = np.array(datas[index])
                self.label = np.array(labels[index])
                pred = self.forward()
                self.backword()
                for i in range(pred.shape[0]):
                    error += (pred[i] - self.Y[i]) * (pred[i] - self.Y[i])
            if epoch % 20 == 0:
                print("epoch... "+epoch+"   error: "+error)

    def forward(self):
        # 从输入层到隐藏层
        self.H = np.dot(self.X, self.W)
        self.H = [self.sigmod(i) for i in self.H]

        # 从隐藏层到输出层
        self.Y = np.dot(self.H, self.V)
        self.Y = [self.sigmod(i) for i in self.Y]
        return self.Y

    def backword(self):
        # 输出层到隐藏层更新V
        for j in range(self.V.shape[1]):
            # 计算每个y的误差值
            g_j = -(self.label[j] - self.Y[j]) * self.Y[j] * (1 - self.Y[j])
            for i in range(self.V.shape[0]):
                self.V[i][j] = g_j * self.H[i]
        # 隐藏层到输入层更新W
        for j in range(self.W.shape[1]):
            delta = 0
            for k in range(self.Y.shape[0]):
                g_j = -(self.label[j] - self.Y[j]) * self.Y[j] * (1 - self.Y[j])
                delta += g_j * self.V[j][k]
            for i in range(self.W.shape[0]):
                self.W[i][j] = delta * self.X[i]

    def sigmod(self, x):
        return 1.0 / (1.0 - np.exp(-x))

    def predict(self, data):
        self.X = data
        return self.forward()

