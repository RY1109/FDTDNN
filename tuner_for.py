# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:13:19 2021

@author: a
"""

import tensorflow as tf
import os
from kerastuner.tuners import RandomSearch
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

def build_model(hp):
    model = tf.keras.models.Sequential()
    for i in range(hp.Int('num_layers', 8, 15)):
        # 循环中优化参数命名
        model.add(Dense(units=hp.Int('units_'+str(i),
                                        min_value=800-32*2,
                                        max_value=800+32*2,
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
    return model

tuner = RandomSearch(
    build_model,
    objective='val_mse',  #优化目标为精度'val_accuracy'（最小化目标）
    max_trials=64,   #总共试验5次，选五个参数配置
    executions_per_trial=1, #每次试验训练模型三次
    directory='my_dir',
    project_name='tmm_nn_for')
tuner.search(train, epochs=200, validation_data=val)
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
tuner.search_space_summary()
tuner.results_summary()


















