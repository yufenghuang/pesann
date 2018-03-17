# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:55:05 2018

@author: hif10
"""

import numpy as np
import tensorflow as tf
import tf_func as tff
import md 
import np_func as npf

params={
    "task":1,
    "chunkSize": 0,
    "epoch": 5000,
    "restart": False,
    "inputData": "MOVEMENT.train.first100",
    "featFile": "feat",
    "engyFile": "engy",
    "logDir": "log",
    "iGPU": 0,
    "dcut": 6.2,
    "learningRate": 0.0001,
    "n2bBasis": 100,
    "n3bBasis": 10,
    "nL1Nodes": 300,
    "nL2Nodes": 500,
    "validate": -1,    #1: calculate the validation after every epoch
    "test": -1,  #0: only calculate the errors on the data set a the end
    "validationSet": "",
    "testSet": "",
    "feRatio": 1.0,
    "dt": 1.0, #picosecond
    "nstep":100, # print every 100 steps
    }

params["restart"]=True
params["epoch"]=100
params["dt"]=0.02
params["nstep"]=10


paramFile = str(params['logDir'])+"/params"
loadParams = np.load(paramFile+".npz")
params["duct"] = float(loadParams["dcut"])
params["n2bBasis"] = int(loadParams["n2bBasis"])
params["n3bBasis"] = int(loadParams["n3bBasis"])
params["nL1Nodes"] = int(loadParams["nL1Nodes"])
params["nL2Nodes"] = int(loadParams["nL2Nodes"])
params["featScalerA"] = loadParams["featScalerA"]
params["featScalerB"] = loadParams["featScalerB"]
params["engyScalerA"] = loadParams["engyScalerA"]
params["engyScalerB"] = loadParams["engyScalerB"]

numFeat = params['n2bBasis'] + params['n3bBasis']**3

tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,float(params['dcut']))
tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)

#tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
#tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)
#
#tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
#tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
#
#
#
#tf_feats = tff.tf_getFeatsFromR(tfCoord, tfLattice, float(params['dcut']), params['n2bBasis'],params['n3bBasis'])*tfFeatA+tfFeatB
#
#tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
#tfEp = (tfEs - tfEngyB) / tfEngyA
#tfFp = tfFs / tfEngyA
#
#saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
with open(params["inputData"], 'r') as mmtFile:
    nAtoms, lattice, R, V0 = md.getRVmmt(mmtFile)

#with tf.Session() as sess:
sess = tf.Session()

fd = {tfCoord: R, tfLattice: lattice}
sess.run(tf.global_variables_initializer())

Rh = sess.run(tfRhat, feed_dict=fd)

#saver.restore(sess, str(params['logDir']) + "/tf.chpt")
#Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
#Fp = -Fp
#
#feedDict = {tfCoord: R, tfLattice:lattice}
#feat = sess.run(tf_feats,feed_dict=feedDict)
#e2 = sess.run(tff.tf_engyFromFeats(tf_feats, numFeat, int(params['nL1Nodes']), int(params['nL2Nodes'])), feed_dict=feedDict)
#e2 = (e2-params['engyScalerB'])/params['engyScalerA']
#
#outNP = npf.np_getNb(R, lattice, params['dcut'])
#outTF = sess.run(tff.tf_getNb(tfCoord, tfLattice, params['dcut']), feed_dict=feedDict)
