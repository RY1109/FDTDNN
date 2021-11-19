# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 19:30:47 2021

@author: a
"""

from tensorflow.keras.models import Model
import tensorflow as tf
from U_get_data import Data_processing 
import os
import pandas as pd
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.autograph.experimental.do_not_convert
tf.keras.backend.set_floatx('float16')
from basicblock import Up,Down

class U_net(Model):
    def __init__(self):
        super(U_net,self).__init__()
        
        self.down1 = Down(channels=64)
        self.down2 = Down(channels=128)
        self.down3 = Down(channels=256)
        self.down4 = Down(channels=512,dropout=0.5)
        
        self.bottom = Down(channels=1024,dropout=0.5,pool_size=(1,1))
        
        self.up1 = Up(channels=512)
        self.up2 = Up(channels=256)
        self.up3 = Up(channels=128)
        self.up4 = Up(channels=89)
    def call(self,x):
        [c1,x] = self.down1(x)
        [c2,x] = self.down2(x)
        [c3,x] = self.down3(x)
        [c4,x] = self.down4(x)
        [c5,x] = self.bottom(x)
        x = self.up1(x,c4)
        x = self.up2(x,c3)
        x = self.up3(x,c2)
        x = self.up4(x,c1)
        return x




class MyUnet(object):
    def __init__(self, img_rows=512, img_cols=512, num=10):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num = num
    def load_data(self):
        dp = Data_processing(num=self.num)
        mydata = dp.creat_data_set()
        return mydata
    def get_unet(self,lr=0.001,decay_batch=20,rate=0.5,staircase=True):
        model=U_net()
        exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=lr, decay_steps=decay_batch, decay_rate=rate,staircase=True)
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
        path = "./checkpoint/U_test2/"
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
        
        self.history = model.fit(train, epochs=10, 
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
        data = self.load_data()
        test=data[2]
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('../data_set/unet.hdf5')
        print('predict test data')
        predict = model.predict(test, batch_size=1, verbose=1)
        # np.save('../data_set/imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    unet = MyUnet(num=4)
    unet.get_unet()
    unet.train()































    
if __name__=='__main__':
    u=U_net()