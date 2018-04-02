# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tf_func as tff

import matplotlib.pyplot as plt

import pandas as pd

###########################################
# 
#    Cosine & its derivative confirmed
#
###########################################

'''
x = np.linspace(-1.2,1.2,200)

tfX = tf.placeholder(tf.float64, shape=(None))

tfY = tff.tf_getCos(tfX, 10)
tfZ = tff.tf_getdCos(tfX, 10)

sess = tf.Session()

y,z = sess.run((tfY, tfZ), feed_dict={tfX: x})

y1 = y[:,1]
z1 = z[:,1]

#plt.plot(x,y1,x,z1)

y1d = (y1[2:] - y1[:-2])/(x[2:] - x[:-2])
plt.plot(x[1:-1], y1d,x,z1,'o')
'''


###########################################
# 
#    Rhat: no problem found
#
###########################################

###########################################
# 
#    dXi/dRl, and dEl/dXi
#
###########################################

df = pd.read_csv('pd_out.csv', header=None)

x = df.iloc[:,0].values
El = df.iloc[:,1].values
Xi = df.iloc[:,2:1102].values
dXi = df.iloc[:,1102:4402].values
dEdXi = df.iloc[:,4402:].values

dXi = dXi.reshape((30,1100,3))[:,:,0]


for n in np.arange(3,10):
    plt.figure()
    Xin = Xi[:,n]
    Xind = (Xin[2:] - Xin[:-2])/(x[2:] - x[:-2])
    plt.plot(x[1:-1], Xind,'-o', x, dXi[:,n], x, dXi[:,n]/2, '-.')
    
    
for n in np.arange(112,118):
    plt.figure()
    Xin = Xi[:,n]
    Xind = (Xin[2:] - Xin[:-2])/(x[2:] - x[:-2])
    plt.plot(x[1:-1], Xind,'-o', x, dXi[:,n], x, dXi[:,n]/2, '-.')