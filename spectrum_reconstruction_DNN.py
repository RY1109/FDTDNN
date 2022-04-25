# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:47:29 2021

@author: a
"""

import scipy.io as sc 
from mytmm import tmm_initial as tmmi
from numpy import inf
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense,LeakyReLU,BatchNormalization, Activation ,Dropout
from tensorflow.keras import Model
import tensorflow as tf
import os
tf.compat.v1.enable_eager_execution()
tf.autograph.experimental.do_not_convert
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.keras.backend.set_floatx('float32')
np.set_printoptions(threshold=np.inf)
def load_data(size): 
    import scipy.io as sc
    val_path = "./balloons_ms/val"
    val_name = os.listdir(val_path)
    val = np.zeros([26215*size,89])
    for i in range(size):   
        data = sc.loadmat(val_path+'/'+val_name[i])
        val[i*26215:(i+1)*26215,:] = data['val']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return val
class ZjuBaseline(Model):
    def __init__(self):
        super(ZjuBaseline, self).__init__()
        self.c1 = Dense(200)  # 卷积层
        self.b1 = BatchNormalization(momentum=0.1, epsilon=1e-5)  # BN层
        self.a1 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c2 = Dense(800)  # 卷积层
        self.b2 = BatchNormalization(momentum=0.1, epsilon=1e-5)  # BN层
        self.a2 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c3 = Dense(800)  # 卷积层
        self.d3 = Dropout(0.1)
        self.b3 = BatchNormalization(momentum=0.1, epsilon=1e-5)  # BN层
        self.a3 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c4 = Dense(800)  # 卷积层
        self.d4 = Dropout(0.1)
        self.b4 = BatchNormalization(momentum=0.1, epsilon=1e-5)  # BN层
        self.a4 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c5 = Dense(800)  # 卷积层
        self.d5 = Dropout(0.1)
        self.b5 = BatchNormalization(momentum=0.1, epsilon=1e-5)  # BN层
        self.a5 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c6 = Dense(100)  # 卷积层
        self.a6 = Activation('sigmoid')  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.c3(x)
        x = self.d3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.c5(x)
        x = self.d5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        y = self.a6(x)
        return x
class ZjuModel(Model):
    def __init__(self):
        super(ZjuModel, self).__init__()
        self.a0 = LeakyReLU(name='a0',alpha=1e-2)
        # self.d1 = Dense(500,
        #                   kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        # self.a1 = LeakyReLU(name='a1',alpha=1e-2)  # 激活层
        # self.d2 = Dense(500,
        #           kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        # self.a2 = LeakyReLU(name='a1',alpha=1e-2)  # 激活层
        self.d3 = Dense(500,name='d1')  # 卷积层
        self.a3 = LeakyReLU(name='a1',alpha=1e-2)  # 激活层
        self.d4 = Dense(500,name='d2')  # 卷积层
        self.a4 = LeakyReLU(name='a2',alpha=1e-2)  # 激活层
        self.d5 = Dense(89,name='d3')  # 卷积层
        self.a5 = LeakyReLU(name='a3',alpha=1e-2)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
    def call(self, x):
        x = self.a0(x)
        # x = self.d1(x)
        # x = self.a1(x)
        # x = self.d2(x)
        # x = self.a2(x)
        x = self.d3(x)
        x = self.a3(x)
        x = self.d4(x)
        x = self.a4(x)
        x = self.d5(x)
        y = self.a5(x)
        return y
class Zjumodel(Model):
    def __init__(self):
        super(Zjumodel, self).__init__()
        self.a0 = LeakyReLU(name='a0')
        # self.d1 = Dense(500,
        #                 kernel_regularizer=tf.keras.regularizers.l2(0), name='d1')  # 卷积层
        # self.a1 = LeakyReLU(name='a1', alpha=1e-2)  # 激活层
        # self.d2 = Dense(500,
        #                 kernel_regularizer=tf.keras.regularizers.l2(0), name='d1')  # 卷积层
        # self.a2 = LeakyReLU(name='a1', alpha=1e-2)  # 激活层
        # self.d3 = Dense(500,
        #                 kernel_regularizer=tf.keras.regularizers.l2(0), name='d1')  # 卷积层
        self.a3 = LeakyReLU(name='a1', alpha=1e-2)  # 激活层
        self.d4 = Dense(500,
                        kernel_regularizer=tf.keras.regularizers.l2(0), name='d2')  # 卷积层
        self.a4 = LeakyReLU(name='a2', alpha=1e-2)  # 激活层
        self.d5 = Dense(89, name='d3')  # 卷积层
        self.a5 = LeakyReLU(name='a3', alpha=1e-2)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
        self.model = ZjuBaseline()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
                           loss='mse',
                           metrics=['me', 'mae'])
        self.checkpoint_save_path = "./checkpoint/Baseline_zjumodel_mydata_drop/Baseline.ckpt"
        self.model.load_weights(self.checkpoint_save_path)
        for layer in self.model.layers:
            layer.trainable = False

    def call(self, x):
        W = self.model(self.P)[:, :89]
        x = tf.transpose(x)
        x = tf.matmul(W, x)
        x = tf.transpose(x)
        x = self.a0(x)
        # x = self.d1(x)
        # x = self.a1(x)
        # x = self.d2(x)
        # x = self.a2(x)
        x = self.d3(x)
        x = self.a3(x)
        x = self.d4(x)
        x = self.a4(x)
        x = self.d5(x)
        y = self.a5(x)
        return y
size = 10
numTF = 4
path = "./checkpoint/DNN_zjubaseline_zjumodel_4_2s/"
checkpoint_save_path = path + "checkpoint.ckpt"
para = sc.loadmat(path+'para.mat')
para = para['paraments']
d_list = [inf, 100,200,100,200,100,200, 200,200,200,200,inf]
ran = np.array(range(numTF))
T_array = np.ones([numTF,89])
val=load_data(size)   
model = ZjuModel()
base = ZjuBaseline()
base.load_weights( "./checkpoint/Baseline_zjumodel_mydata_drop4/Baseline.ckpt")
for i in tqdm(ran):
    lambda_lis  = np.array(range(100))*4+400
    inpu = para[i,:]*1000
    inpu = inpu.reshape(10,).tolist()
    d_list[1:11] = inpu
    [lambda_list,T_list,_] = tmmi.sample2(d_list,89)
    # T_list2 = base.predict(para[i,:].reshape([1,10]))
    # plt.plot(lambda_list, T_list2[0,:])
    # plt.plot(lambda_list,T_list)
    # plt.show()
    T_array[i,:] = np.array(T_list).reshape([1,89])

test_input = np.matmul(val, T_array.T)

model.load_weights(checkpoint_save_path)

predict = model.predict(test_input)


result = plt.figure() ##像素点光谱
result_ = result.add_subplot(1,1,1)
ran = np.array(range(10))
for i in tqdm(ran):
    result = plt.figure() ##像素点光谱
    result_ = result.add_subplot(1,1,1)
    result_.plot(np.array(range(89)).reshape(89,1),predict[i,:],label='Pre')
    result_.plot(np.array(range(89)).reshape(89,1),val[i,:],label='True')
    result_.legend()
    result_.set(title=str(np.sum(np.square(predict[i,:]-val[i,:]))/89))
    plt.show()
    # result.savefig('./result/figure/ballons_DNN/'+str(i))
