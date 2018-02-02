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
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,float(params["dcut"]))
    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/params["dcut"]-2,params['n2bBasis']),[tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/params["dcut"]-2,params['n3bBasis']),[tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/params["dcut"]-2,params['n3bBasis']),[tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    tfFeats = tff.tf_getFeats(tfGR2,tfGR3,tfGD3)
    tfFeats = tfFeats * tfFeatA + tfFeatB
    tfEs = tff.tf_engyFromFeats(tfFeats, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])
    tfEi = (tfEs - tfEngyB)/tfEngyA
    
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with tf.Session() as sess:
        with open(params["mmtFile"], 'r') as mmtFile:
            nAtoms, lattice, R= getRmmt(mmtFile)
        feedDict={
                tfCoord:R,
                tfLattice: lattice
                }
        sess.run(tf.global_variables_initializer())
        if params["restart"]:
            saver.restore(sess, str(params['logDir'])+"/tf.chpt")
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
    feat = getFeats(R, lattice, float(params['dcut']), params['n2bBasis'],params['n3bBasis'])
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

def outputFeatures(params):
    import os
    import tensorflow as tf
    import tf_func as tff
    import pandas as pd
    
    if os.path.exists(str(params["featFile"])):
        os.remove(str(params["featFile"]))
    if os.path.exists(str(params["engyFile"])):
        os.remove(str(params["engyFile"]))
        
    tfR = tf.placeholder(tf.float32, shape=(None,3))
    tfL = tf.placeholder(tf.float32, shape=(3,3))
    
    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1
    
    sess = tf.Session()
    with open(params["inputData"], 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = getData(datafile)
            feedDict = {tfR: R, tfL:lattice}
            feat = sess.run(tff.tf_getFeatsFromR(tfR,tfL,float(params['dcut']), params['n2bBasis'],params['n3bBasis']), feed_dict=feedDict)
            engy = e.reshape([-1,1])
            pd.DataFrame(feat).to_csv(params["featFile"],mode='a',header=False,index=False)
            pd.DataFrame(engy).to_csv(params["engyFile"],mode='a',header=False,index=False)

def trainEF(params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)
    
    # Tensorflow placeholders
    tfCoord = tf.placeholder(tf.float32, shape=(None,3))
    tfLattice = tf.placeholder(tf.float32, shape=(3,3))
    tfEngy = tf.placeholder(tf.float32, shape=(None))
    tfFors = tf.placeholder(tf.float32, shape=(None,3))
    tfLearningRate = tf.placeholder(tf.float32)
    
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,float(params['dcut']))
    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)
    
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),[tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),[tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),[tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi>0),tff.tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),[tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/float(params['dcut'])-2,params['n3bBasis']),[tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    
    tfdXi, tfdXin = tff.tf_get_dXidRl(tfGR2,tfGR2d,tfGR3,tfGR3d,tfGD3,tfRhat)
    tfdXi =  tf.expand_dims(tfFeatA,2) * tfdXi 
    tfdXin =  tf.expand_dims(tfFeatA,2) * tfdXin
    
    tfFeats = tfFeatA*tff.tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
    tfEs = tff.tf_engyFromFeats(tfFeats, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])
    
    
    dEldXi = tff.tf_get_dEldXi(tfFeats, params['numFeat'], params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi,2)*tfdXi,axis=1)
    
    dENldXi=tf.gather_nd(dEldXi,tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb,0))-1),1))
    dEnldXin=tf.scatter_nd(tf.where(tf.greater(tfIdxNb,0)), dENldXi, [tfNAtoms,tfMaxNb,params['numFeat']])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*tfdXin,axis=[1,2])
    
    tfFs = Fln + Fll 
    
    tfEp = (tfEs - tfEngyB)/tfEngyA
    tfFp = tfFs/tfEngyA
    
    tfLoss = tf.reduce_mean(tf.squared_difference(tfEs, tfEngy)) + \
             float(params['feRatio']) * tf.reduce_mean(tf.squared_difference(tfFs, tfFors))
             
    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
            tfOptimizer = tf.train.AdamOptimizer(tfLearningRate).minimize(tfLoss)
             
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    if params["restart"]:
        saver.restore(sess, str(params['logDir'])+"/tf.chpt")
        print("Model restored")
    
    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1
    
    for iEpoch in range(params["epoch"]):
        file = open(str(params["inputData"]), 'r')
        for iCase in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(file)    
            feedDict={
                    tfCoord:R,
                    tfLattice: lattice,
                    tfEngy: e*params['engyScalerA']+params['engyScalerB'], 
                    tfFors: f*params['engyScalerA'],
                    tfLearningRate: float(params['learningRate']),
                    }
            sess.run(tfOptimizer,feed_dict=feedDict)
            
#            loss = sess.run(tfLoss, feed_dict=feedDict)
            (Ei,Fi) = sess.run((tfEp,tfFp),feed_dict=feedDict)
            Ermse = np.sqrt(np.mean((Ei-e)**2))
            Frmse = np.sqrt(np.mean((Fi-f)**2))
            print(iEpoch, "Ermse:", Ermse)
            print(iEpoch, "Frmse:", Frmse)
    
        file.close()
        
#        loss = sess.run(tfLoss, feed_dict=feedDict)
        (Ei,Fi) = sess.run((tfEp,tfFp),feed_dict=feedDict)
        Ermse = np.sqrt(np.mean((Ei-e)**2))
        Frmse = np.sqrt(np.mean((Fi-f)**2))
        print(iEpoch, "Ermse:", Ermse)
        print(iEpoch, "Frmse:", Frmse)
        
    save_path = saver.save(sess, str(params['logDir'])+"/tf.chpt")
    return save_path