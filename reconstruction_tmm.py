# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:25:53 2021

@author: a
"""


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation ,LeakyReLU ,Dropout
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tqdm import tqdm
from mytmm import tmm_initial as tmmi
from numpy import inf
import scipy.io as sc
def load_data(): 
    import scipy.io as sc
    import numpy as np
    name = "./mytmm/data/test_10layers.mat"
    data = sc.loadmat(name)
    T = data['T']
    d = data['d']*0.001
    return [d,T]
[train,label] = load_data()  

np.random.seed(116)
np.random.shuffle(train)
np.random.seed(116)
np.random.shuffle(label)

data_num = np.size(train,0)
test_input=train[int(0.7*data_num):,:]
test_label=label[int(0.7*data_num):,:]


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

class Zju(Model):
    def __init__(self):
        
        super(Zju, self).__init__()
        self.c1 = Dense(200)  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = LeakyReLU()  # 激活层
        self.c2 = Dense(800)  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = LeakyReLU()  # 激活层
        self.c3 = Dense(800)  # 卷积层
        self.d3 = Dropout(0.1)
        self.b3 = BatchNormalization()  # BN层
        self.a3 = LeakyReLU()  # 激活层
        self.c4 = Dense(800)  # 卷积层
        self.d4 = Dropout(0.1)
        self.b4 = BatchNormalization()  # BN层
        self.a4 = LeakyReLU()  # 激活层
        self.c5 = Dense(800)  # 卷积层
        self.d5 = Dropout(0.1)        
        self.b5 = BatchNormalization()  # BN层
        self.a5 = LeakyReLU()  # 激活层
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
        return y
 


model = Zju()

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001),
              loss='mse',
              metrics=['mse','mae'])

path = "./checkpoint/Baseline_zjumodel_mydata/"
checkpoint_save_path = path + "Baseline.ckpt"
model_save_path = path + "Baseline.tf"

model.load_weights(checkpoint_save_path)
predict = model.predict(test_input)
d_list = [inf, 100,200,100,200,100,200, 200,200,200,200,inf]

result = plt.figure() ##像素点光谱
result_ = result.add_subplot(1,1,1)
ran = np.array(range(16)) 
for i in tqdm(ran):
    lambda_lis  = np.array(range(100))*4+400
    inpu = test_input[i,:]*1000
    inpu = inpu.reshape(10,).tolist()
    d_list[1:11] = inpu
    [lambda_list,T_list] = tmmi.sample2(d_list,1000)
    result = plt.figure() ##像素点光谱
    result_ = result.add_subplot(1,1,1)
    result_.plot(lambda_list, T_list,label='tmm')
    result_.plot(lambda_lis,predict[i,:],label='pre')
    
    # a=sc.loadmat('T_test'+str(i+1))['T'][0,0]['T']
    # test_label=np.squeeze(a).reshape([100,1])
    # result_.plot(np.array(range(100)).reshape(100,1),test_label[i,:],label='true')
    
    
    
    result_.legend()
    result_.imshow
#     result.savefig('./result/figure/filter16/'+str(i))
#     sc.savemat('./result/data/filters16/'+str(i)+'.mat',{'tmm':T_list,'pre':predict[i,:]})
# sc.savemat('./result/data/filters16/para.mat',{'paraments':test_input})
