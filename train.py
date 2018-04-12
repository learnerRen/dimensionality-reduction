# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:42:13 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
import model
import numpy as np
import time
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
model_save_path="./path/model/"
model_name="model.ckpt"
batch_size=128
epoch=10000
def train():
    t0=time.time()
    x=tf.placeholder(tf.float32,
                     [None,model.n0],
                     name='x-input')
    y=tf.placeholder(tf.float32,
                     [None,model.n4])
    y_code,y_hat=model.inference(x)
    global_step=tf.Variable(0,trainable=False)
    mse=tf.square(y-y_hat)
    reduce_mse=tf.reduce_mean(mse,name='mse')
    loss=reduce_mse+tf.add_n(tf.get_collection('loss'))
    train_op=tf.train.AdamOptimizer().minimize(loss,global_step=global_step)
    saver=tf.train.Saver()
    q1=0
    q2=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            xs,_=mnist.train.next_batch(batch_size)
            _,step=sess.run([train_op,global_step],feed_dict={x:xs,y:xs})
            if step%2000==0:
                loss_value,step=sess.run([loss,global_step],feed_dict={x:xs,y:xs})
                print("After {} steps, training loss:{}".format(step-1,loss_value))
            if i==epoch-1:
                y_code_value=sess.run([y_code],feed_dict={x:mnist.train.images[0:2000,:]})
                y_code_value=np.array(y_code_value)
                y_code_value=y_code_value.reshape(2000,2)
                np.savetxt('y_code.txt',y_code_value,delimiter=',')
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
    t1=time.time()
    return t1-t0,q1,q2
t,q1,q2=train()
print("training time:{}".format(t))