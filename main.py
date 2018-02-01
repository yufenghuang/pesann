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

##############################################################################
#
#   Initializing the program
#
##############################################################################

def initialize(params):
    params['dcut'] = 6.2
    params['learningRate']=0.0001
    params['n2bBasis'] = 100
    params['n3bBasis'] = 10
    params['numFeat'] = params['n2bBasis'] + params['n3bBasis'] **3
    params['nL1Nodes'] = 300
    params['nL2Nodes'] = 500
    
    file = open(str(params["inputData"]), 'r')
    nAtoms, iIter, lattice, R, f, v, e = pyf.getData(file)
    feat = pyf.getFeats(R, lattice, params['dcut'], params['n2bBasis'],params['n3bBasis'])
    engy = e.reshape([-1,1])
    file.close()
    
    featScalerA,featScalerB,engyScalerA,engyScalerB = pyf.getFeatEngyScaler(feat,engy)
    
    params['featScalerA'],params['featScalerB'],params['engyScalerA'],params['engyScalerB'] = \
    pyf.getFeatEngyScaler(feat,engy)
    
#    feat_scaled = params['featScalerA'] * feat + params['featScalerB']
#    engy_scaled = params['engyScalerA'] * engy + params['engyScalerB']
    
    if not os.path.exists(str(params['logDir'])):
        os.mkdir(str(params['logDir']))
            
    paramFile = str(params['logDir'])+"/params"
    np.savez(paramFile,**params)
    return params

params = {
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
        "nL2Nodes": 500
        }

os.environ["CUDA_VISIBLE_DEVICES"]=str(params['iGPU'])

if params['restart']:
    paramFile = str(params['logDir'])+"/params"
    loadParams = np.load(paramFile+".npz")
    for param in loadParams.files:
        params[param] = loadParams[param]
        
else:
    params = initialize(params)

print("Initialization done")
    
'''
if os.path.exists(featFile):
    os.remove(featFile)
if os.path.exists(engyFile):
    os.remove(engyFile)
tfR = tf.placeholder(tf.float32, shape=(None,3))
tfL = tf.placeholder(tf.float32, shape=(3,3))

nCase = 0
with open(inputData, 'r') as datafile:
    for line in datafile:
        if "Iteration" in line:
            nCase += 1

sess = tf.Session()
with open(inputData, 'r') as datafile:
    for i in range(nCase):
        nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
        feedDict = {tfR: R, tfL:lattice}
        feat = sess.run(tff.tf_getFeatsFromR(tfR,tfL,dcut,n2bBasis, n3bBasis), feed_dict=feedDict)
        engy = e.reshape([-1,1])
        pd.DataFrame(feat).to_csv(featFile,mode='a',header=False,index=False)
        pd.DataFrame(engy).to_csv(engyFile,mode='a',header=False,index=False)
'''


##############################################################################
#
#   Defining the Neural Network
#
##############################################################################

# Tensorflow constants
tf_pi = tf.constant(np.pi, dtype=tf.float32)
tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

# train with features

tfFeat = tf.placeholder(tf.float32,shape=(None, params['numFeat']))
tfEngy = tf.placeholder(tf.float32,shape=(None, 1))
tfLR = tf.placeholder(tf.float32)

tfEs = tff.tf_engyFromFeats(tfFeat, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])

tfLoss = tf.reduce_mean((tfEs-tfEngy)**2)

with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
    tfOptimizer = tf.train.AdamOptimizer(tfLR).minimize(tfLoss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if params['chunkSize'] == 0:
    dfFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False).values
    dfEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False).values
    
for iEpoch in range(params['epoch']):
    if params['chunkSize'] > 0:
        pdFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False, \
                             chunksize=int(params['chunkSize']), iterator=True)
        pdEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False, \
                             chunksize=int(params['chunkSize']), iterator=True)
        
        for pdF in pdFeat:
            pdE = next(pdEngy)
            dfFeat = pdF.values
            dfEngy = pdE.values
            feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'], \
                        tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'], \
                        tfLR: params['learningRate']}
        
            sess.run(tfOptimizer, feed_dict=feedDict)
            print("running",iEpoch)

    elif params['chunkSize'] == 0:
        feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'], \
                    tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'], \
                    tfLR: params['learningRate']}
    
        sess.run(tfOptimizer, feed_dict=feedDict)
    else:
        print("Invalid chunkSize, not within [0,inf]. chunkSize=",params['chunkSize'])

    if iEpoch % 10 == 0:
        Ep, loss = sess.run((tfEs, tfLoss), feed_dict=feedDict)
        Ep = (Ep - params['engyScalerB'])/params['engyScalerA']
        Ermse = np.sqrt(np.mean((Ep - dfEngy)**2))
        print(iEpoch, loss, Ermse)

saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
save_path = saver.save(sess, str(params['logDir'])+"/tf.chpt")



'''
# Tensorflow placeholders
tfCoord = tf.placeholder(tf.float32, shape=(None,3))
tfLattice = tf.placeholder(tf.float32, shape=(3,3))
tfEngy = tf.placeholder(tf.float32, shape=(None))
tfFors = tf.placeholder(tf.float32, shape=(None,3))
tfLearningRate = tf.placeholder(tf.float32)

tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,dcut)
tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)

tfGR2 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n2bBasis),[tfNAtoms,tfMaxNb,n2bBasis])
tfGR2d = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n2bBasis),[tfNAtoms,tfMaxNb,n2bBasis])
tfGR3 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n3bBasis),[tfNAtoms,tfMaxNb,n3bBasis])
tfGR3d = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n3bBasis),[tfNAtoms,tfMaxNb,n3bBasis])
tfGD3 = tf.scatter_nd(tf.where(tfDc>0),tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/dcut-2,n3bBasis),[tfNAtoms,tfMaxNb, tfMaxNb,n3bBasis])

tfdXi, tfdXin = tff.tf_get_dXidRl(tfGR2,tfGR2d,tfGR3,tfGR3d,tfGD3,tfRhat)
tfdXi =  tf.expand_dims(tfFeatA,2) * tfdXi 
tfdXin =  tf.expand_dims(tfFeatA,2) * tfdXin

tfFeats = tfFeatA*tff.tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
tfEs = tff.tf_engyFromFeats(tfFeats, numFeat, nL1Nodes, nL2Nodes)


dEldXi = tff.tf_get_dEldXi(tfFeats, numFeat, nL1Nodes, nL2Nodes)
Fll = tf.reduce_sum(tf.expand_dims(dEldXi,2)*tfdXi,axis=1)

dENldXi=tf.gather_nd(dEldXi,tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb,0))-1),1))
dEnldXin=tf.scatter_nd(tf.where(tf.greater(tfIdxNb,0)), dENldXi, [tfNAtoms,tfMaxNb,numFeat])
Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*tfdXin,axis=[1,2])

tfFs = Fln + Fll 

#tfEp = (tfEs-tfEngyB)/tfEngyA
#tfFp = tfFs/tfEngyA

#loss = tf.reduce_mean(tf.squared_difference(prediction,Y)) #+ tf.reduce_sum(regLoss)
#lossEF = tf.reduce_mean(tf.squared_difference(prediction,Y)) + \
#        tf.reduce_mean(tf.squared_difference(predictedF,tfForces))
#optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer2=tf.train.AdamOptimizer(learning_rate, name="lossEF").minimize(lossEF)

#tfFeatures = tff.tf_getFeats(tfGR2,tfGR3,tfGD3)


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