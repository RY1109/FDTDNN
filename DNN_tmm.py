# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:38:09 2021

@author: a
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.autograph.experimental.do_not_convert
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation ,LeakyReLU ,Dropout
from tensorflow.keras import Model
import tensorflow as tf
import scipy.io as sc
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.autograph.experimental.do_not_convert
tf.keras.backend.set_floatx('float32')
np.set_printoptions(threshold=np.inf)



def load_data(size): 
    import scipy.io as sc
    train_path = "./balloons_ms/train"
    test_path = "./balloons_ms/test"
    train_name = os.listdir(train_path)
    test_name = os.listdir(test_path)
    train = np.zeros([157286*size,89])
    test = np.zeros([78644*size,89])
    for i in range(size):   
        data = sc.loadmat(train_path+'/'+train_name[i])
        train[i*157286:(i+1)*157286,:] = data['train']
        # train = (train - np.min(train))/ (np.max(train)-np.min(train))
        data = sc.loadmat(test_path+'/'+test_name[i])
        test[i*78644:(i+1)*78644,:] = data['test']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return [train,test]


def Diy_loss(labels, predictions,P,
             delta=0.01,betar=1e-3):
    mse = tf.reduce_mean(tf.square(labels-predictions))
    l = tf.constant(np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]),dtype='float32')
    u = tf.constant(np.array([[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]),dtype='float32')
    regu = tf.maximum(tf.add(P, tf.add(-u, delta)), 
                      tf.maximum(tf.add(-P, tf.add(l, delta)),
                                 0))/delta
    loss = betar*(tf.reduce_mean(regu))+mse   
    return loss
    


def Diy_loss2(labels, predictions,P,
             delta=0.01,betar=0.001):
    mse = tf.reduce_mean(tf.square(labels-predictions))
    l = tf.constant(np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]),dtype='float32')
    m = tf.constant(np.array([[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]]),dtype='float32')
    u = tf.constant(np.array([[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]),dtype='float32')
    # regu = tf.maximum(tf.abs(tf.add(-u, P)), 
    #                   tf.maximum(tf.abs(tf.add(P, -m)),
    #                              tf.maximum(tf.abs(tf.add(P,-l)),
    #                                         0)))
    regu = tf.abs(tf.multiply(  tf.multiply(tf.add(P,-l), tf.add(P,-m)), tf.add(P,-u)))
    loss = betar*(tf.reduce_mean(regu))+mse   
    return loss

    
class Mybaseline(Model):
    def __init__(self):
        super(Mybaseline, self).__init__()
        self.c1 = Dense(40)  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.c3 = Dense(80)  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活层
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

class Zjubaseline(Model):
    def __init__(self):
        
        super(Zjubaseline, self).__init__()
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
     
    
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.P = self.add_weight(
            shape=(16, 10), initializer="random_normal", trainable=True,name='P'
        )
        self.d1 = Dense(10,
                          kernel_regularizer=tf.keras.regularizers.l2(0),name='d1')  # 卷积层
        self.a1 = Activation('relu')  # 激活层
        self.d5 = Dense(89,name='d2')  # 卷积层
        self.a5 = Activation(None)  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
        self.model = Mybaseline()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001),
              loss='mse',
              metrics=['me','mae'])
        self.checkpoint_save_path = "./checkpoint/Baseline_zjumodel_mydata/Baseline.ckpt"
        self.model.load_weights(self.checkpoint_save_path)
        for layer in model.layers:
            layer.trainable = False
    def call(self, x):
        W = self.model(self.P)[:,:89]
        x = tf.transpose(x) 
        x = tf.matmul(W,x)
        x = tf.transpose(x) 
        x = self.d1(x)
        x = self.a1(x)
        x = self.d5(x)
        y = self.a5(x)
        return y
    
    
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
                          kernel_regularizer=tf.keras.regularizers.l2(0),name='d2')  # 卷积层
        self.a2 = LeakyReLU(name='a2')  # 激活层
        self.d3 = Dense(89,name='d3')  # 卷积层
        self.a3 = LeakyReLU(name='a3')  # 激活层x_temp = data_all[0,0][mode + '_train'][:,0:x_len,0:T]
        self.model = Zjubaseline()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001,decay=0.0001),
              loss='mse',
              metrics=['me','mae'])
        self.checkpoint_save_path = "./checkpoint/Baseline_zjumodel_mydata/Baseline.ckpt"
        self.model.load_weights(self.checkpoint_save_path)
        for layer in self.model.layers:
            layer.trainable = False
    def call(self, x):
        W = self.model(self.P)[:,:89]
        x = tf.transpose(x) 
        x = tf.matmul(W,x)
        x = tf.transpose(x) 
        # x = self.a0(x)
        x = self.d1(x)
        x = self.a1(x)
        x = self.d2(x)
        y = self.a2(x)
        x = self.d3(x)
        y = self.a3(x)
        return y

size = 10    
model = ZjuModel()
path = "./checkpoint/DNN_zjubaseline_zjumodel_10datasets/"
checkpoint_save_path = path + "checkpoint.ckpt"
model_save_path = path + "checkpoint.tf"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path).expect_partial()

  
# loss_object =    tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr = 1e-4,decay=0.001)

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')


[train,test] = load_data(size)   
# train = train[:500,:]
# test = test[:500,:]

@tf.function
def train_step(tra):
  with tf.GradientTape() as tape:
    predictions = model(tra)
    loss = Diy_loss(tra, predictions,model.P)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(tra,predictions)
  train_accuracy(tra, predictions)

@tf.function
def test_step(tes):
  predictions = model(tes)
  t_loss = Diy_loss(tes, predictions,model.P)

  test_loss(tes,predictions)
  test_accuracy(tes, predictions)

EPOCHS = 1
ltr = len(train)
lte = len(test)
batch_size=2000
loss = np.zeros([4,EPOCHS])

for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for index in range(0,ltr,batch_size):
    tra = train[index:(index+batch_size),:]
    if len(tra)<batch_size:
        break
    tra=tf.constant(tra.reshape([batch_size,89]),dtype='float32')
    train_step(tra)

  for index in range(0,lte,batch_size):
    tes = test[index:(index+batch_size),:]
    if len(tes)<batch_size:
        break
    tes=tf.constant(tes.reshape([batch_size,89]),dtype='float32')
    test_step(tes)
    
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result(),
                         test_loss.result(),
                         test_accuracy.result()))
  
  loss[0,epoch]=train_loss.result().numpy()
  loss[1,epoch]=train_accuracy.result().numpy()
  loss[2,epoch]=test_loss.result().numpy()
  loss[3,epoch]=test_accuracy.result().numpy()
  
model.summary()
# print(model.trainable_variables)
file = open(path+'weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
model.save_weights(checkpoint_save_path)  
model.save(model_save_path) 
test_input = model.P.numpy()
sc.savemat(path+'para.mat',{'paraments':test_input})

def plot_history(loss):
   
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(np.array(range(EPOCHS)), loss[0,:],
           label='Train Error')
  plt.plot(np.array(range(EPOCHS)), loss[2,:],
           label='Val Error')
  plt.legend()
  plt.show()


plot_history(loss)


















