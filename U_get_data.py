# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:01:39 2021

@author: a
"""
import numpy as np
import tensorflow as tf
import scipy.io as sc
import os
class Data_processing():
    def __init__(self,size=[512,512,89],data_path='./balloons_ms/U_net/',num=1,name='hyper2'):
        self.data_path = data_path
        self.num = num
        self.size=size
        self.train = np.zeros([num,size[0],size[1],size[2]])
        self.name=name
    def load_data(self):
        path_list=os.listdir(self.data_path)
        for index, path in enumerate(path_list):
            if index >=self.num:
                break
            self.train[index,:,:,:] = sc.loadmat(self.data_path + path)[self.name]
        return self.train
    def creat_data_set(self):
        train = self.load_data()
        train = train.astype('float32')
        train_examples = train[:int(self.num*0.6),:,:,:]
        train_labels = train_examples
        val_examples = train[int(self.num*0.6):int(self.num*0.9),:,:,:]
        val_labels = val_examples
        test_examples = train[int(self.num*0.9):,:,:,:]
        test_labels = test_examples
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
        return [train_dataset,val_dataset,test_dataset]
if __name__=='__main__':
    a=Data_processing()
    b=a.creat_data_set()
