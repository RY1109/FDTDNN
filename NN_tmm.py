# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:13:19 2021

@author: a
"""


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model




np.set_printoptions(threshold=np.inf)
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)




def load_data(): 
    import scipy.io as sc
    import numpy as np
    name = "./tmm/data/test.mat"
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
test_input=train[:int(0.7*data_num),:]
test_label=label[:int(0.7*data_num),:]





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


model = Baseline()

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01,decay=0.0001),
              loss='mse',
              metrics=['mse','mae'])

checkpoint_save_path = "./checkpoint/Baseline_tmm/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = ([ tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                          save_weights_only=True,
                                                          save_best_only=True),
                    tf.keras.callbacks.EarlyStopping(patience=100, min_delta=1e-4)
                   ])

history = model.fit(train, label, batch_size=32, epochs=300, validation_split=0.2, validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./checkpoint/Baseline_tmm/weights.txt', 'w')
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

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
  plt.legend()
  plt.show()


plot_history(history)



















