# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:50:40 2021

@author: a
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf 

def load_data():
    import scipy.io as sc
    name = "./balloons_ms/balloons_val"
    data = sc.loadmat(name)
    val = data['val']
    return val


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Dense(40)  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.c3 = Dense(80)  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活层
        # self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.c2 = Dense(100)  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation(None)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        # x = self.p1(x)
        # x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        y = self.a2(x)
        return y


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.P = self.add_weight(
            shape=(4, 2), initializer="random_normal", trainable=True,name='P'
        )
        self.d1 = Dense(10,
                          kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        # self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        # self.d2 = Dense(100,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0))  # 卷积层
        # self.b2 = BatchNormalization()  # BN层
        # self.a2 = Activation('relu')  # 激活层
        # self.d4 = Dense(100,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0))  # 卷积层
        # self.b4 = BatchNormalization()  # BN层
        # self.a4 = Activation('relu')  # 激活层
        # self.d3 = Dense(500,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0))  # 卷积层
        # self.b3 = BatchNormalization()  # BN层
        # self.a3 = Activation('relu')  # 激活层
        # self.d4 = Dense(100,
        #                  kernel_regularizer=tf.keras.regularizers.l2(0))  # 卷积层
        # self.b4 = BatchNormalization()  # BN层
        # self.a4 = Activation('relu')  # 激活层
        # self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d5 = Dense(89,name='d2')  # 卷积层
        # self.b5 = BatchNormalization()  # BN层
        self.a5 = Activation(None)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
        self.model = Baseline()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001),
              loss='mse',
              metrics=['me','mae'])

        self.checkpoint_save_path = "./checkpoint/Baseline2/Baseline.ckpt"
        self.model.load_weights(self.checkpoint_save_path)
    def call(self, x):
        W = self.model(self.P)[:,:89]
        x = tf.transpose(x) 
        x = tf.matmul(W,x)
        x = tf.transpose(x) 
        x = self.d1(x)
        # x = self.b1(x)
        x = self.a1(x)
        # x = self.d2(x)
        # x = self.b2(x)
        # x = self.a2(x)
        # x = self.d3(x)
        # x = self.b3(x)
        # x = self.a3(x)
        # x = self.d4(x)
        # x = self.b4(x)
        # x = self.a4(x)
        # x = self.p1(x)
        # x = self.d1(x)
        x = self.d5(x)
        # x = self.b5(x)
        y = self.a5(x)
        return y
    
model=MyModel()
val=load_data()
checkpoint_save_path='./checkpoint/DNN3/checkpoint'

model.load_weights(checkpoint_save_path)
predict = model.predict(val)


result = plt.figure() ##像素点光谱
result_ = result.add_subplot(1,1,1)
ran = np.array(range(200)) 
for i in tqdm(ran):
    result = plt.figure() ##像素点光谱
    result_ = result.add_subplot(1,1,1)
    result_.plot(np.array(range(89)).reshape(89,1),predict[i,:],label='Pre')
    result_.plot(np.array(range(89)).reshape(89,1),val[i,:],label='True')
    result_.legend()
    result_.imshow
    result.savefig('./figure/ballons_2/'+str(i))