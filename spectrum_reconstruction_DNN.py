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
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.autograph.experimental.do_not_convert
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.layers import Dense,LeakyReLU
from tensorflow.keras import Model
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.autograph.experimental.do_not_convert
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
class ZjuModel(Model):
    def __init__(self):
        super(ZjuModel, self).__init__()
        self.P = self.add_weight(
            shape=(16, 10), initializer="random_normal", trainable=True,name='P'
        )
        # self.a0 = LeakyReLU(name='a0')
        self.d1 = Dense(500,
                          kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        self.a1 = LeakyReLU(name='a1')  # 激活层
        self.d2 = Dense(500,
                  kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        self.a2 = LeakyReLU(name='a1')  # 激活层
        self.d3 = Dense(500,
                  kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        self.a3 = LeakyReLU(name='a1')  # 激活层
        self.d4 = Dense(500,
                          kernel_regularizer=tf.keras.regularizers.l2(0),name='d2')  # 卷积层
        self.a4 = LeakyReLU(name='a2')  # 激活层
        self.d5 = Dense(89,name='d3')  # 卷积层
        self.a5 = LeakyReLU(name='a3')  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
    def call(self, x):
        x = self.d1(x)
        x = self.a1(x)
        x = self.d2(x)
        x = self.a2(x)
        x = self.d3(x)
        x = self.a3(x)
        x = self.d4(x)
        x = self.a4(x)
        x = self.d5(x)
        y = self.a5(x)
        return y


size = 10  
path = "./checkpoint/DNN_zjubaseline_zjumodel_16filter__/"
checkpoint_save_path = path + "checkpoint.ckpt"
para = sc.loadmat(path+'para.mat')
para = para['paraments']
d_list = [inf, 100,200,100,200,100,200, 200,200,200,200,inf]
ran = np.array(range(16)) 
T_array = np.ones([16,89])
val=load_data(size)   
model = ZjuModel()



for i in tqdm(ran):
    lambda_lis  = np.array(range(100))*4+400
    inpu = para[i,:]*1000
    inpu = inpu.reshape(10,).tolist()
    d_list[1:11] = inpu
    [lambda_list,T_list] = tmmi.sample2(d_list,89)
    plt.plot(lambda_list,T_list)
    plt.show()
    T_array[i,:] = np.array(T_list).reshape([1,89])

test_input = np.matmul(val, T_array.T)

model.load_weights(checkpoint_save_path)
predict = model.predict(test_input)


result = plt.figure() ##像素点光谱
result_ = result.add_subplot(1,1,1)
ran = np.array(range(20)) 
for i in tqdm(ran):
    result = plt.figure() ##像素点光谱
    result_ = result.add_subplot(1,1,1)
    result_.plot(np.array(range(89)).reshape(89,1),predict[i,:],label='Pre')
    result_.plot(np.array(range(89)).reshape(89,1),val[i,:],label='True')
    result_.legend()
    result_.set(title=str(np.sum(np.square(predict[i,:]-val[i,:]))/89))
    result_.imshow
    # result.savefig('./result/figure/ballons_DNN/'+str(i))
