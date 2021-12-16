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
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, LeakyReLU
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
    import numpy as np
    name = "./mytmm/data/test_10layers"
    data = sc.loadmat(name)
    T = data['T']
    d = data['d'] * 0.001
    return [d, T]


[train, label] = load_data()
data_num = np.size(train, 0)
train = train[:int(0.7 * data_num), :]
label = label[:int(0.7 * data_num), :]
dataset = tf.data.Dataset.from_tensor_slices((train, label))
train = dataset.take(int(0.6 * data_num))
val = dataset.skip(int(0.6 * data_num))

BATCH_SIZE = 2048
SHUFFLE_BUFFER_SIZE = 100
train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val = val.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential()
for i in []:
    # 循环中优化参数命名
    model.add(Dense(units=hp.Int('units_' + str(i),
                                 min_value=800 - 32 * 2,
                                 max_value=800 + 32 * 2,
                                 step=32),
                    activation=None))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())  # BN层
    model.add(LeakyReLU())  # 激活层
model.add(Dense(100, activation='sigmoid'))
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=200 * 2048, decay_rate=0.8, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
              loss='mse',
              metrics=['mse', 'mae'])

exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=200 * 2048, decay_rate=0.8, staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
              loss='mse',
              metrics=['mse', 'mae'])
path = "./checkpoint/Baseline_zjumodel_mydata____/"
checkpoint_save_path = path + "Baseline.ckpt"
model_save_path = path + "Baseline.tf"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = ([tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
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
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    plt.show()


plot_history(history)




















