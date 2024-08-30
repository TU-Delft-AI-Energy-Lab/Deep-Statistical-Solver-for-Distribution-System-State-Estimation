# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:30:52 2022

@author: Benjamin Habib
"""

import numpy as np
import tensorflow as tf
import os

folder = "data_ober2"

case_name = "meas1"


case = {}


case[0] = "bmeas1"
case[1] = "meas1"
case[2] = "gmeas1"


u = {}
a = {}
b = {}

c = 3

n = 1

for j in range(c):
    for i in range(n):
    
        u[j*n + i] = np.load(folder+'/U_'+case[j]+'('+str(i)+').npy', allow_pickle=True)
        a[j*n + i] = np.load(folder+'/A_'+case[j]+'('+str(i)+').npy', allow_pickle=True)
        b[j*n + i] = np.load(folder+'/B_'+case[j]+'('+str(i)+').npy', allow_pickle=True)
  
A = a[0]
B= b[0]
U = u[0]

for i in range(1,c*n):
        
    A = np.concatenate((A, a[i]))
    B = np.concatenate((B, b[i]))
    U = np.concatenate((U, u[i]))
    
ind = np.arange(A.shape[0])
np.random.shuffle(ind)

sample_ratio = 0.8

case = case_name

path_train=os.path.join(folder,'train_'+case+'.tfrecords')
path_val=os.path.join(folder,'val_'+case+'.tfrecords')
path_test=os.path.join(folder,'test_'+case+'.tfrecords')

path_all = os.path.join(folder,'data_'+case+'.tfrecords')

    
with tf.io.TFRecordWriter(path_train) as file_writer:
  for i in range(int(np.ceil(A.shape[0]) * sample_ratio)):
      
    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        "A": tf.train.Feature(float_list=tf.train.FloatList(value=A[ind[i]].flatten().tolist())),
        "B": tf.train.Feature(float_list=tf.train.FloatList(value=B[ind[i]].flatten().tolist())),
        "U": tf.train.Feature(float_list=tf.train.FloatList(value=U[ind[i]].flatten().tolist())),
    })).SerializeToString()
    file_writer.write(record_bytes)
    
with tf.io.TFRecordWriter(path_val) as file_writer:
  for i in range(int(np.ceil(A.shape[0] * sample_ratio)),
                  int(np.ceil(A.shape[0] * (0.5 + 0.5 * sample_ratio)))):
      
    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        "A": tf.train.Feature(float_list=tf.train.FloatList(value=A[ind[i]].flatten().tolist())),
        "B": tf.train.Feature(float_list=tf.train.FloatList(value=B[ind[i]].flatten().tolist())),
        "U": tf.train.Feature(float_list=tf.train.FloatList(value=U[ind[i]].flatten().tolist())),
    })).SerializeToString()
    file_writer.write(record_bytes)

with tf.io.TFRecordWriter(path_test) as file_writer:
  for i in range(int(np.ceil(A.shape[0] * (0.5 + 0.5 * sample_ratio))), A.shape[0]):
      
    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        "A": tf.train.Feature(float_list=tf.train.FloatList(value=A[ind[i]].flatten().tolist())),
        "B": tf.train.Feature(float_list=tf.train.FloatList(value=B[ind[i]].flatten().tolist())),
        "U": tf.train.Feature(float_list=tf.train.FloatList(value=U[ind[i]].flatten().tolist())),
    })).SerializeToString()
    file_writer.write(record_bytes)




