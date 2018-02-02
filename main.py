#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:49:17 2018

@author: yufeng
"""
import tensorflow as tf
import numpy as np
import pandas as pd

import tf_func as tff
import py_func as pyf

import os

#import re
##############################################################################
#
#   Initializing the program
#
##############################################################################

'''
sampleInput = "# comments \n" \
              "a = 1 \n" \
              "b = \"movement file\" # specifying b \n" \
              "c = # wrong input \n"

for line in sampleInput.split("\n"):
    line0 = line.split('#')[0].strip()
    if line0 != '':
        if re.match("^([\w\s\"]+)=([\w\s\"]+)$", line0):
            key = line0.split("=")[0].strip().strip("\"")
            value = line0.split("=")[1].strip().strip("\"")
            print(key,"is",value)
        else:
            print("unknown input", line0)
'''
params = {
        "chunkSize": 0,
        "epoch": 10,
        "restart": True,
        "inputData": "MOVEMENT.train.first100",
        "featFile": "feat",
        "engyFile": "engy",
        "logDir": "log",
        "iGPU": 0,
        "runtype": -2,   # 2: evaluate energy and forces
                        # 1: evaluate energy
                        # 0: MD
                        #-1: training with energy
                        #-2: training with energy and forces
        "mmtFile": "coord.mmt",
        "feRatio": 1,
        }

os.environ["CUDA_VISIBLE_DEVICES"]=str(params['iGPU'])

if params['restart']:
    paramFile = str(params['logDir'])+"/params"
    loadParams = np.load(paramFile+".npz")
    oldParams = {}
    for param in loadParams.files:
        oldParams[param] = loadParams[param]
    for param in params:
        oldParams[param] = params[param]
    params = oldParams
else:
    params = pyf.initialize(params)

print("Initialization done")

if params["runtype"] == 2:
    pass
elif params["runtype"] == 1:
    params["mmtFile"] = "coord.mmt"
    Ep = pyf.getEngy(params)
elif params["runtype"] == 0:
    pass
elif params["runtype"] == -1:
    params["featFile"] = "feat"
    params["engyFile"] = "engy"
    pyf.outputFeatures(params)
    print(pyf.trainEngy(params))
elif params["runtype"] == -2:
    print(pyf.trainEF(params))
else:
    print("Unrecognized runtype: ", params["runtype"])
    

##############################################################################
#
#   Defining the Neural Network
#
##############################################################################

# Tensorflow constants
#tf_pi = tf.constant(np.pi, dtype=tf.float32)
#tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
#tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
#tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
#tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

# train with features



#tfEp = (tfEs-tfEngyB)/tfEngyA
#tfFp = tfFs/tfEngyA

#loss = tf.reduce_mean(tf.squared_difference(prediction,Y)) #+ tf.reduce_sum(regLoss)
#lossEF = tf.reduce_mean(tf.squared_difference(prediction,Y)) + \
#        tf.reduce_mean(tf.squared_difference(predictedF,tfForces))
#optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer2=tf.train.AdamOptimizer(learning_rate, name="lossEF").minimize(lossEF)

#tfFeatures = tff.tf_getFeats(tfGR2,tfGR3,tfGD3)

'''
##############################################################################
#
#   Evaluating the Neural Network
#
##############################################################################


sess = tf.Session()
sess.run(tf.global_variables_initializer())

lattice = np.array([[0.2169434000E+02, 0, 0], \
                    [0, 0.1481809000E+02, 0], \
                    [0,0,0.1578275000E+02]])

df = pd.read_csv("Rtemp", header=None, index_col=False, delim_whitespace=True)
R = df.iloc[:,1:4].values

feedDict = {tfCoord: R, tfLattice:lattice}
tmp1 = sess.run(dEnldXin,feed_dict=feedDict)
'''