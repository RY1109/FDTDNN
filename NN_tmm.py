# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:13:19 2021

@author: a
"""


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation ,Dropout,LeakyReLU
from tensorflow.keras import Model
tf.autograph.experimental.do_not_convert
tf.keras.backend.set_floatx('float32')
np.set_printoptions(threshold=np.inf)




def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def load_data():
    import scipy.io as sc
    name = "./mytmm/data/test_10layers.mat"
    data = sc.loadmat(name)
    T = data['T']
    d = data['d']*0.001
    return [d,T]

# def load_data():
#     import scipy.io as sc
#     import numpy as np
#     name = "./mytmm/data/data_TF_100-300nm"
#     data = sc.loadmat(name)
#     T = data['Trans_train'].T
#     d = data['Thick_train'].T*0.001
#     return [d,T]

[train,label] = load_data()
data_num = np.size(train,0)
train=train[:int(0.7*data_num),:]
label=label[:int(0.7*data_num),:]
dataset = tf.data.Dataset.from_tensor_slices((train, label))
train = dataset.take(int(0.6*data_num))
val = dataset.skip(int(0.6*data_num))

BATCH_SIZE = 2048
SHUFFLE_BUFFER_SIZE = 100
train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val = val.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

class Mybaseline(Model):
    def __init__(self):
        
        super(Mybaseline, self).__init__()
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
        self.b1 = BatchNormalization(momentum=0.1,epsilon=1e-5)  # BN层
        self.a1 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c2 = Dense(800)  # 卷积层
        self.b2 = BatchNormalization(momentum=0.1,epsilon=1e-5)  # BN层
        self.a2 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c3 = Dense(800)  # 卷积层
        self.d3 = Dropout(0.1)
        self.b3 = BatchNormalization(momentum=0.1,epsilon=1e-5)  # BN层
        self.a3 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c4 = Dense(800)  # 卷积层
        self.d4 = Dropout(0.1)
        self.b4 = BatchNormalization(momentum=0.1,epsilon=1e-5)  # BN层
        self.a4 = LeakyReLU(alpha=1e-2)  # 激活层
        self.c5 = Dense(800)  # 卷积层
        self.d5 = Dropout(0.1)
        self.b5 = BatchNormalization(momentum=0.1,epsilon=1e-5)  # BN层
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
    
    
model = Zju()
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.001, decay_steps=200*2048, decay_rate=0.8,staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay,epsilon=1e-8),
              loss=tf.keras.losses.MSE,
              metrics=['mse','mae'])
path = "./checkpoint/Baseline_zjumodel_mydata_drop4/"
checkpoint_save_path = path + "Baseline.ckpt"
model_save_path = path + "Baseline.tf"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = ([ tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                          save_weights_only=True,
                                                          save_best_only=True),
                    tf.keras.callbacks.EarlyStopping(patience=2000, min_delta=1e-5)
                   ])

history = model.fit(train, epochs=2000, validation_data=val, validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
model.save(model_save_path)

# print(model.trainable_variables)
file = open(path + 'weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
  plt.legend()
  plt.savefig(path + 'mae')

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
  plt.legend()
  plt.savefig(path + 'mse')
  plt.show()


plot_history(history)




















