# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:07:52 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
n0=784
#layer1
n1=500
#layer2
n2=2
#layer3
n3=500
#layer4
n4=784
def inference(input_tensor):
    regularizer=tf.contrib.layers.l2_regularizer(0.1)
    with tf.variable_scope('layer1'):
        w1=tf.get_variable('weights',
                           [n0,n1],
                           dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer())
        b1=tf.get_variable('bais',
                           [n1],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01))
        tf.add_to_collection('losses',regularizer(w1))
        a1=tf.matmul(input_tensor,w1)+b1
        z1=tf.nn.relu(a1)
    with tf.variable_scope('layer2'):
        w2=tf.get_variable('weights',
                           [n1,n2],
                           dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer())
        b2=tf.get_variable('bais',
                           [n2],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01))
        a2=tf.matmul(z1,w2)+b2
        z2=tf.nn.relu(a2)
    with tf.variable_scope('layer3'):
        w3=tf.get_variable('weights',
                            [n2,n3],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b3=tf.get_variable('bias',
                           [n3],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01))
        tf.add_to_collection('loss',regularizer(w3))
        a3=tf.matmul(z2,w3)+b3
        z3=tf.nn.relu(a3)
    with tf.variable_scope('layer4'):
        w4=tf.get_variable('weight',
                           [n3,n4],
                           dtype=tf.float32,
                           initializer=tf.contrib.layers.variance_scaling_initializer())
        b4=tf.get_variable('bais',
                           [n4],
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01))
        tf.add_to_collection('loss',regularizer(w4))
        a4=tf.matmul(z3,w4)+b4
        z4=tf.nn.relu(a4)
    return a2,z4