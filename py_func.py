# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:59:01 2018

@author: hif10
"""

import re
import numpy as np

def getData(dataFile):
    line = dataFile.readline()
    if "Iteration" in line:
        sptline = line.split()
        nAtoms = int(sptline[0])
        iIter = int(re.match("(\d*),",sptline[2])[1])
        
        dataFile.readline()
        lattice = np.zeros((3,3))
        lattice[0,:] = np.array(dataFile.readline().split(),dtype=float)
        lattice[1,:] = np.array(dataFile.readline().split(),dtype=float)
        lattice[2,:] = np.array(dataFile.readline().split(),dtype=float)
        
        dataFile.readline()
        R = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            R[i,:] = np.array(sptline[1:4])
        
        dataFile.readline()
        forces = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            forces[i,:] = np.array(sptline[1:4])
            
        dataFile.readline()
        velocities = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            velocities[i,:] = np.array(sptline[1:4])
            
        dataFile.readline()
        energies = np.zeros((nAtoms))
        for i in range(nAtoms):
            sptline = dataFile.readline().split()
            energies[i] = float(sptline[1])

        return nAtoms, iIter, lattice, R, forces, velocities,energies 
    
    else:
        return getData(dataFile)

def getRmmt(mmtFile):
    line = mmtFile.readline()
    
    sptline = line.split()
    nAtoms = int(sptline[0])
    
    mmtFile.readline()
    lattice = np.zeros((3,3))
    lattice[0,:] = np.array(mmtFile.readline().split(),dtype=float)
    lattice[1,:] = np.array(mmtFile.readline().split(),dtype=float)
    lattice[2,:] = np.array(mmtFile.readline().split(),dtype=float)
    
    mmtFile.readline()
    R = np.zeros((nAtoms,3))
    for i in range(nAtoms):
        sptline = mmtFile.readline().split()
        R[i,:] = np.array(sptline[1:4])
    
    return nAtoms, lattice, R 
    

def getFeats(R, lattice, dcut,n2bBasis, n3bBasis):
    import tensorflow as tf
    import tf_func as tff
    
    tfCoord = tf.placeholder(tf.float32, shape=(None,3))
    tfLattice = tf.placeholder(tf.float32, shape=(3,3))
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,dcut)
    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n2bBasis),[tfNAtoms,tfMaxNb,n2bBasis])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n3bBasis),[tfNAtoms,tfMaxNb,n3bBasis])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/dcut-2,n3bBasis),[tfNAtoms,tfMaxNb, tfMaxNb,n3bBasis])
    tfFeats = tff.tf_getFeats(tfGR2,tfGR3,tfGD3)

    feedDict={
            tfCoord:R,
            tfLattice: lattice
            }
    
    with tf.Session() as sess:
        feats = sess.run(tfFeats, feed_dict=feedDict)
    return feats

def getFeatEngyScaler(feat,engy):
    from sklearn.preprocessing import MinMaxScaler
    engy_scaler = MinMaxScaler(feature_range=(0,1))
    feat_scaler = MinMaxScaler(feature_range=(0,1))
    
    feat_scaler.fit_transform(feat)
    engy_scaler.fit_transform(engy)
    
    (a,b) = feat.shape
    
    feat_b = feat_scaler.transform(np.zeros((1,b)))
    feat_a = feat_scaler.transform(np.ones((1,b))) - feat_b
    engy_b = engy_scaler.transform(0)
    engy_a = engy_scaler.transform(1) - engy_b
    
    return feat_a,feat_b,engy_a,engy_b
    

def trainEngy(params):
    import tensorflow as tf
    import tf_func as tff
    import pandas as pd
    tfFeat = tf.placeholder(tf.float32,shape=(None, params['numFeat']))
    tfEngy = tf.placeholder(tf.float32,shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)
    
    tfEs = tff.tf_engyFromFeats(tfFeat, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])
    
    tfLoss = tf.reduce_mean((tfEs-tfEngy)**2)
    
    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.AdamOptimizer(tfLR).minimize(tfLoss)
    
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if params["restart"]:
        saver.restore(sess, str(params['logDir'])+"/tf.chpt")
        print("Model restored")
    
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
    
    
    save_path = saver.save(sess, str(params['logDir'])+"/tf.chpt")
    return save_path

def getEngy(params):
    import tensorflow as tf
    import tf_func as tff
    
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)

    tfCoord = tf.placeholder(tf.float32, shape=(None,3))
    tfLattice = tf.placeholder(tf.float32, shape=(3,3))
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,params["dcut"])
    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/params["dcut"]-2,params['n2bBasis']),[tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/params["dcut"]-2,params['n3bBasis']),[tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/params["dcut"]-2,params['n3bBasis']),[tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    tfFeats = tff.tf_getFeats(tfGR2,tfGR3,tfGD3)
    tfFeats = tfFeats * tfFeatA + tfFeatB
    tfEs = tff.tf_engyFromFeats(tfFeats, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])
    tfEi = tfEs * tfEngyA + tfEngyB
    
    with tf.Session() as sess:
        nAtoms, lattice, R= getData(params["mmtFile"])
        feedDict={
                tfCoord:R,
                tfLattice: lattice
                }
        Ei = sess.run(tfEi, feed_dict=feedDict)
    return Ei

def initialize(params):
    import os
    oldParams={
        "chunkSize": 0,
        "epoch": 5000,
        "restart": True,
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
    
    for param in params:
        oldParams[param] = params[param]
        
    params = oldParams
    
    file = open(str(params["inputData"]), 'r')
    nAtoms, iIter, lattice, R, f, v, e = getData(file)
    feat = getFeats(R, lattice, params['dcut'], params['n2bBasis'],params['n3bBasis'])
    engy = e.reshape([-1,1])
    file.close()
    
    featScalerA,featScalerB,engyScalerA,engyScalerB = getFeatEngyScaler(feat,engy)
    
    params['featScalerA'],params['featScalerB'],params['engyScalerA'],params['engyScalerB'] = \
    getFeatEngyScaler(feat,engy)
    
#    feat_scaled = params['featScalerA'] * feat + params['featScalerB']
#    engy_scaled = params['engyScalerA'] * engy + params['engyScalerB']
    
    if not os.path.exists(str(params['logDir'])):
        os.mkdir(str(params['logDir']))
            
    paramFile = str(params['logDir'])+"/params"
    np.savez(paramFile,**params)
    return params
