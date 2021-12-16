# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:32:40 2021

@author: a
"""

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.autograph.experimental.do_not_convert
# tf.compat.v1.disable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tqdm import tqdm

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


def Diy_loss(labels, predictions,P,
             delta=0.01,betar=0.0001):
    mse = tf.reduce_mean(tf.square(labels-predictions))
    l = tf.constant(np.array([[0.1,0.05]]),dtype='float32')
    u = tf.constant(np.array([[0.2,0.2]]),dtype='float32')
    regu = tf.maximum(tf.add(P, tf.add(-u, delta)), 
                      tf.maximum(tf.add(-P, tf.add(l, delta)),
                                 0))/delta
    loss = betar*(tf.reduce_mean(regu))+mse   
    return loss
    
    # def huber_loss(labels, predictions, delta=1.0):
    # residual = tf.abs(predictions - labels)
    # condition = tf.less(residual, delta)
    # small_res = 0.5 * tf.square(residual)
    # large_res = delta * residual - 0.5 * tf.square(delta)
    # return tf.where(condition, small_res, large_res)

    
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
    
model = MyModel()
checkpoint_save_path='./checkpoint/DNN3/checkpoint.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path).expect_partial()

  
# loss_object =    tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr = 0.001)

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')


[train,test] = load_data()   
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

EPOCHS = 5
ltr = len(train)
lte = len(test)
batch_size=32


for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for index in range(0,ltr,batch_size):
    tra = train[index:(index+batch_size),:]
    if len(tra)<32:
        break
    tra=tf.constant(tra.reshape([batch_size,89]),dtype='float32')
    train_step(tra)

  for index in range(0,lte,batch_size):
    tes = test[index:(index+batch_size),:]
    if len(tes)<32:
        break
    tes=tf.constant(tes.reshape([batch_size,89]),dtype='float32')
    test_step(tes)
    
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result(),
                         test_loss.result(),
                         test_accuracy.result()))
model.summary()
# print(model.trainable_variables)
file = open('./checkpoint/DNN3/weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
model.save_weights('./checkpoint/DNN3/checkpoint')   









# model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.001),
#               loss='mse',
#               metrics=['mse','mae'])

# checkpoint_save_path = "./checkpoint/DNN/Mymodel.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)

# cp_callback = ([ tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                           save_weights_only=True,
#                                                       save_best_only=True),
#                 tf.keras.callbacks.EarlyStopping(patience=4, min_delta=1e-4)
#                    ])

# history = model.fit(train, train, batch_size=32, epochs=100, 
#                     validation_data=[test,test], validation_freq=1,
#                     callbacks=[cp_callback])
# # model.summary()

# # print(model.trainable_variables)
# # file = open('./checkpoint/DNN/weights.txt', 'w')
# # for v in model.trainable_variables:
# #     file.write(str(v.name) + '\n')
# #     file.write(str(v.shape) + '\n')
# #     file.write(str(v.numpy()) + '\n')
# # file.close()

# ###############################################    show   ###############################################

# # 显示训练集和验证集的acc和loss曲线
# def plot_history(history):
#   hist = pd.DataFrame(history.history)
#   hist['epoch'] = history.epoch

#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean Abs Error [MPG]')
#   plt.plot(hist['epoch'], hist['mae'],
#             label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mae'],
#             label = 'Val Error')
#   plt.legend()

#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean Square Error [$MPG^2$]')
#   plt.plot(hist['epoch'], hist['mse'],
#             label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mse'],
#             label = 'Val Error')
#   plt.legend()
#   plt.show()


# plot_history(history)




# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

















