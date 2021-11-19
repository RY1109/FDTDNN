# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:55:26 2021

@author: a
"""
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout 
from tensorflow.keras.models import Model


class Down(Model):
    def __init__(self,channels=128, pool_size=(2,2), ker_size=3, stride=1, padding='same', 
                 bias=True,activation='relu',dropout=0,ker_init='he_normal'):
        super(Down,self).__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.ker_size = ker_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
        self.ker_init =ker_init

        self.conv1 = Conv2D(self.channels, self.ker_size, activation=self.activation, 
                            padding=self.padding, kernel_initializer=self.ker_init)
        self.conv2 = Conv2D(self.channels, self.ker_size, activation=self.activation, 
                            padding=self.padding, kernel_initializer=self.ker_init)
        self.drop1 = Dropout(self.dropout)
        self.max = MaxPooling2D(pool_size=self.pool_size)
    def call(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.drop1(x)
        x2 = self.max(x1)
        return [x1,x2]


class Up(Model):
    def __init__(self,channels=64, pool_size=(2,2), ker_size=3, stride=1, padding='same', 
                 bias=True,activation='relu',dropout=0,ker_init='he_normal'):
        super(Up,self).__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.ker_size = ker_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
        self.ker_init =ker_init
        
        self.up=UpSampling2D(size=self.pool_size)
        self.conv1 = Conv2D(self.channels, self.ker_size, activation=self.activation, 
                            padding=self.padding, kernel_initializer=self.ker_init)
        self.concat = Concatenate(axis=3)
        self.conv2 = Conv2D(self.channels, self.ker_size, activation=self.activation, 
                            padding=self.padding, kernel_initializer=self.ker_init)
        self.conv3 = Conv2D(self.channels, self.ker_size, activation=self.activation, 
                    padding=self.padding, kernel_initializer=self.ker_init)
    def call(self,x,conlayer):
        x = self.up(x)
        x = self.conv1(x)
        x = self.concat([x,conlayer])
        x = self.conv2(x)
        x = self.conv3(x)
        return x

if __name__ == '__main__':
    model=Down(channels=1000)
    model2=Down(channels=20)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    