# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:32:40 2021

@author: a
"""

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tqdm import tqdm
from kerastuner.tuners import RandomSearch


def load_data(): 
    import scipy.io as sc
    name = "./balloons_ms/balloons_train"
    data = sc.loadmat(name)
    train = data['train']
    # train = (train - np.min(train))/ (np.max(train)-np.min(train))
    name = "./balloons_ms/balloons_test"
    data = sc.loadmat(name)
    test = data['test']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return [train,test]






class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Dense(40)  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        # self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.c2 = Dense(100)  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation(None)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]

    def call(self, x, training=None, mask=None, **kwargs):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        # x = self.p1(x)
        # x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        y = self.a2(x)
        return y
    
    
def build_model(hp):     
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.P = self.add_weight(
                shape=(4, 2), initializer="random_normal", trainable=True,name='P'
            )
            self.c1 = Dense(40,
                             kernel_regularizer=tf.keras.regularizers.l2(0))  # 卷积层
            self.b1 = BatchNormalization()  # BN层
            self.a1 = Activation('relu')  # 激活层
            # self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
            self.c2 = Dense(89)  # 卷积层
            self.b2 = BatchNormalization()  # BN层
            self.a2 = Activation(None)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
            self.model = Baseline()
    
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001),
                  loss='mse',
                  metrics=['mse','mae'])
    
            self.checkpoint_save_path = "./checkpoint/Baseline.ckpt"
            self.model.load_weights(self.checkpoint_save_path)
        def call(self, x):
            W = self.model.call(self.P,training=False)[:,:89]
            x = tf.transpose(x) 
            x = tf.matmul(W,x)
            x = tf.transpose(x) 
            x = self.c1(x)
            x = self.b1(x)
            x = self.a1(x)
            # x = self.p1(x)
            # x = self.d1(x)
    
            x = self.c2(x)
            x = self.b2(x)
            y = self.a2(x)
            return y
    model = MyModel()
    
    model.compile(optimizer=tf.keras.optimizers.Adam
                  (hp.Choice('lr',values=[0.01,0.001,0.1,0.0001]),
                   hp.Choice('decay',values=[0.1,0.001,0.0001,0.01])),
                  loss='mse',
                  metrics=['mse','mae'])
    return model

[train,test] = load_data()   

tuner = RandomSearch(
    build_model,
    objective='val_mse',  #优化目标为精度'*'（最小化目标）
    max_trials=16,   #总共试验*次，选*个参数配置
    executions_per_trial=3, #每次试验训练模型三次
    directory='my_dir',
    project_name='PolonelayerNNlr')
tuner.search(train, train, batch_size=32, epochs=10, validation_data=(test,test))
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
# model.summary()

# print(model.trainable_variables)
# file = open('./checkpoint/DNN/weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线























