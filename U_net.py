# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:14:15 2021

@author: a
"""


from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout ,Softmax,Input, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from U_get_data import Data_processing 
import os
import pandas as pd
from matplotlib import pyplot as plt

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512, num=10):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num = num
    def load_data(self):
        dp = Data_processing(num=self.num)
        mydata = dp.creat_data_set()

        return mydata
    
    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 89))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='1_1')(inputs)
        print(conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='1_2')(conv1)
        print(conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print(pool1.shape)
        print('\n')

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='2_1')(pool1)
        print(conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='2_2')(conv2)
        print(conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print(pool2.shape)
        print('\n')

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='3_1')(pool2)
        print(conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='3_2')(conv3)
        print(conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print(pool3.shape)
        print('\n')

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='4_1')(pool3)
        print(conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='4_2')(conv4)
        print(conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print(pool4.shape)
        print('\n')

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='5_1')(pool4)
        print(conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='5_2')(conv5)
        print(conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print('\n')

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        print(up6.shape)
        print(drop4.shape)     
        Concatenate6 = Concatenate(axis=3)([drop4, up6])
        print('Concatenate: ')
        print(Concatenate6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        Concatenate7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        Concatenate8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        Concatenate9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Concatenate9)
        conv9 = Conv2D(89, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        conv10 = Activation('sigmoid',name='sm1')(conv9)
        print(conv10.shape)

        model = Model(inputs=inputs, outputs=conv10)
        exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=0.001, decay_steps=200*2048, decay_rate=0.8,staircase=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),
                      loss='mse',
                      metrics=['mse','mae'])
        print('model compile')
        return model
    
    def plot_history(self):
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch
        
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
        
    def train(self):
        data =self.load_data()
        train = data[0]
        val = data[1]
        BATCH_SIZE = 1
        SHUFFLE_BUFFER_SIZE = 1
        train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        val = val.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        model = self.get_unet()
        path = "./checkpoint/Baseline_zjumodel_mydata____/"
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
        
        self.history = model.fit(train, epochs=2000, 
                            validation_data=val, validation_freq=1,
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
        self.plot_history()

    def test(self):
        print("loading data")
        imgs_test = self.load_test_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('../data_set/unet.hdf5')
        print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../data_set/imgs_mask_test.npy', imgs_mask_test)
        
        


if __name__ == '__main__':
    unet = myUnet(num=2)
    unet.get_unet()
    unet.train()