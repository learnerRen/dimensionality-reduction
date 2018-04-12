# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 00:22:26 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

from sklearn.manifold import TSNE
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
tsne=TSNE(n_components=2,init='pca',random_state=0)
X=tsne.fit_transform(mnist.train.images[0:2000,:])
X=np.array(X)
np.savetxt('tsne.txt',X,delimiter=',')