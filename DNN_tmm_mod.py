# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:38:09 2021

@author: a
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation ,LeakyReLU ,Dropout
import U_net
from U_get_data import Data_processing 
from tensorflow.keras import Model
import tensorflow as tf
import scipy.io as sc
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.keras.backend.set_floatx('float32')
np.set_printoptions(threshold=np.inf)

def load_data(num):
    dp = Data_processing(num=num)
    mydata = dp.creat_data_set()
    return mydata


def Diy_loss(labels, predictions,P,
             delta=0.01,betar=0.001):
    mse = tf.reduce_mean(tf.square(labels-predictions),axis=3)
    # print(mse.numpy())
    l = tf.constant(np.array([[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]]),dtype='float32')
    u = tf.constant(np.array([[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]),dtype='float32')
    regu = tf.maximum(tf.add(P, tf.add(-u, delta)), 
                      tf.maximum(tf.add(-P, tf.add(l, delta)),
                                 0))/delta
    loss = betar*(tf.reduce_mean(regu))+mse   
    return loss
        

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
         
class ZjuModel(Model):
    def __init__(self):
        super(ZjuModel, self).__init__()
        self.P = self.add_weight(
            shape=(1, 10), initializer="random_normal", trainable=True,name='P'
        )
        self.U = U_net.U_net()
        self.model = Zjubaseline()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001,decay=0.0001),
              loss='mse',
              metrics=['me','mae'])
        self.checkpoint_save_path = "./checkpoint/Baseline_zjumodel_mydata_100_0.8/Baseline.ckpt"
        self.model.load_weights(self.checkpoint_save_path)
        for layer in self.model.layers:
            layer.trainable = False
    def call(self, x):
        W = self.model(self.P)[:,:89]
        x = tf.matmul(x,W,transpose_b=True)
        # x = self.a0(x)
        y = self.U(x)
        return y
num = 30
model = ZjuModel()
path = "./checkpoint/DNN_zjubaseline_U_net_1thin/"
checkpoint_save_path = path + "checkpoint.ckpt"
model_save_path = path + "checkpoint.tf"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path).expect_partial()

  
# loss_object =    tf.keras.losses.MeanSquaredError()
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.0001, decay_steps=50*2048, decay_rate=0.8,staircase=True)
optimizer = tf.keras.optimizers.Adam(exponential_decay)

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

dp = Data_processing(num=num)
data = dp.creat_data_set()

train = data[0]
val = data[1]
BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 1
train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val = val.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) 
# train = train[:500,:]
# test = test[:500,:]

@tf.function
@tf.autograph.experimental.do_not_convert
def train_step(tra):
  with tf.GradientTape() as tape:
    predictions = model(tra,training=True)
    loss = Diy_loss(tra, predictions,model.P)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(tra,predictions)
  train_accuracy(tra, predictions)

@tf.function
@tf.autograph.experimental.do_not_convert
def test_step(tes):
  predictions = model(tes)
  t_loss = Diy_loss(tes, predictions,model.P)

  test_loss(tes,predictions)
  test_accuracy(tes, predictions)

EPOCHS = 10
loss = np.zeros([4,EPOCHS],dtype='float32')

for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for tra in train:
    train_step(tra[0])

  for va in val:
    test_step(va[0])
    
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
model.save_weights(checkpoint_save_path)  
# model.save(model_save_path) 
test_input = model.P.numpy()
sc.savemat(path+'para.mat',{'paraments':test_input})


file = open(path+'weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

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