# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:59:01 2018

@author: hif10
"""

import re
import numpy as np
import tensorflow as tf
import tf_func as tff
import sys

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
            energies[i] = float(sptline[1])-271

        R[R > 1] = R[R > 1] - np.floor(R[R > 1])
        R[R < 0] = R[R < 0] - np.floor(R[R < 0])

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

    R[R>1] = R[R>1] - np.floor(R[R>1])
    R[R<0] = R[R<0] - np.floor(R[R<0])

    return nAtoms, lattice, R 
    

def getFeats(R, lattice, dcut_in ,n2bBasis, n3bBasis):
#    import tensorflow as tf
#    import tf_func as tff
    
    dcut = tf.constant(dcut_in, dtype=tf.float64)
    tfCoord = tf.placeholder(tf.float64, shape=(None,3))
    tfLattice = tf.placeholder(tf.float64, shape=(3,3))
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tff.tf_getNb(tfCoord,tfLattice,dcut)
    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n2bBasis),\
                          [tfNAtoms,tfMaxNb,n2bBasis])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tff.tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n3bBasis),\
                          [tfNAtoms,tfMaxNb,n3bBasis])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tff.tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/dcut-2,n3bBasis),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,n3bBasis])
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
#    import tensorflow as tf
#    import tf_func as tff
    import pandas as pd
    numFeat = params['n2bBasis'] + params['n3bBasis']**3
    tfFeat = tf.placeholder(tf.float64,shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float64,shape=(None, 1))
    tfLR = tf.placeholder(tf.float64)
    
    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    
    tfLoss = tf.reduce_mean((tfEs-tfEngy)**2)
    
    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.AdamOptimizer(tfLR).minimize(tfLoss)
    
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    def getError(fFile, eFile):
        featDF = pd.read_csv(fFile, header=None, index_col=False).values
        engyDF = pd.read_csv(eFile, header=None, index_col=False).values
        feedDict2 = {tfFeat: featDF * params['featScalerA'] + params['featScalerB'], \
                     tfEngy: engyDF * params['engyScalerA'] + params['engyScalerB']}
        Ep2 = sess.run(tfEs, feed_dict=feedDict2)
        Ep2 = (Ep2 - params['engyScalerB'])/params['engyScalerA']
        
        Ermse = np.sqrt(np.mean((Ep2-engyDF)**2))
        Emae = np.mean(np.abs(Ep2 - engyDF))
        print("Ermse is: ", Ermse)
        print("Emae is : ", Emae)
        sys.stdout.flush()

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
#                print("running",iEpoch)
    
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
            sys.stdout.flush()
            
        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print(str(iEpoch)+"th epoch")
                getError('v'+str(params['featFile']), 'v'+str(params['engyFile']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print(str(iEpoch)+"th epoch")
                getError('t'+str(params['featFile']), 't'+str(params['engyFile']))
        
    if params["validate"] == 0:
        getError('v'+str(params['featFile']), 'v'+str(params['engyFile']))
    if params["test"] == 0:
        getError('t'+str(params['featFile']), 't'+str(params['engyFile']))

    save_path = saver.save(sess, str(params['logDir'])+"/tf.chpt")
    return save_path

def getEngy(params):
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float64)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float64)

    tfCoord = tf.placeholder(tf.float64, shape=(None,3))
    tfLattice = tf.placeholder(tf.float64, shape=(3,3))
    
    tfEs=tff.tf_getE(tfCoord, tfLattice,params)
    tfEi = (tfEs - tfEngyB)/tfEngyA
    
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with tf.Session() as sess:
        with open(params["inputData"], 'r') as mmtFile:
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

def NVE(params):

    J = 1/1.602177e-19 # eV
    meter = 1e10 # Angstroms
    s = 1e12 # ps
    mole = 6.022141e23 # atoms
    kg = 1e3*mole # grams/mole

    mSi = 28.09 # grams/mol

    constA = J/(kg*meter**2/s**2)

    dt = float(params["dt"])

    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float64)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float64)

    tfCoord = tf.placeholder(tf.float64, shape=(None, 3))
    tfLattice = tf.placeholder(tf.float64, shape=(3, 3))

    tfEs, tfFs = tff.tf_getEF(tfCoord, tfLattice, params)
    tfEp = (tfEs - tfEngyB) / tfEngyA
    tfFp = tfFs / tfEngyA

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with open(params["inputData"], 'r') as mmtFile:
        nAtoms, lattice, R = getRmmt(mmtFile)

    R0 = R.dot(lattice.T)
    R1 = np.zeros_like(R0)
    V0 = np.zeros_like(R0)
    Vpos = np.zeros_like(R0)
    Vneg = np.zeros_like(R0)

    with tf.Session() as sess:
        feedDict = {tfCoord: R, tfLattice: lattice}
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
        Fp = -Fp

        Vpos = 0.5*Fp/mSi*dt * constA

        # V0 = Vneg + 0.5*Fp/mSi*dt * constA
        
        R1 = R0 + Vpos * dt
        
        Epot = np.sum(Ep)
        Ekin = np.sum(0.5*mSi*V0**2/constA)
        Etot = Epot + Ekin

        print(nAtoms)
        print(0,"Epot=", "{:.12f}".format(Epot), "Ekin=","{:.12f}".format(Ekin), "Etot=","{:.12f}".format(Etot))
        for iAtom in range(len(R1)):
            print("Si", R1[iAtom, 0], R1[iAtom, 1], R1[iAtom, 2], V0[iAtom, 0], V0[iAtom, 1], V0[iAtom, 2])
        sys.stdout.flush()

        for iStep in range(1,params["epoch"]):
            R0 = R1
            Vneg = Vpos
            R = np.linalg.solve(lattice, R0.T).T

            R[R > 1] = R[R > 1] - np.floor(R[R > 1])
            R[R < 0] = R[R < 0] - np.floor(R[R < 0])
            R0 = R.dot(lattice.T)

            feedDict = {tfCoord: R, tfLattice: lattice}
            Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
            Fp = -Fp
            V0 = Vneg + 0.5*Fp/mSi*dt * constA
            Vpos = Vneg + Fp/mSi*dt * constA
            R1 = R0 + Vpos * dt
            
            Epot = np.sum(Ep)
            Ekin = np.sum(0.5*mSi*V0**2/constA)
            Etot = Epot + Ekin

            if (iStep % int(params["nstep"]) == 0) or \
                ((iStep % int(params["nstep"]) != 0) & (iStep==params["epoch"]-1)):
                print(nAtoms)
                print(iStep,"Epot=", "{:.12f}".format(Epot), "Ekin=","{:.12f}".format(Ekin), "Etot=","{:.12f}".format(Etot))
                for iAtom in range(len(R1)):
                    print("Si",R1[iAtom,0], R1[iAtom,1],R1[iAtom,2], V0[iAtom,0], V0[iAtom,1], V0[iAtom,2])
                sys.stdout.flush()

#            if (iStep % int(params["nstep"]) != 0) & (iStep==params["epoch"]-1):
#                print(nAtoms)
#                print(np.sum(Ep))
#                for iAtom in range(len(R1)):
#                    print("Si", R1[iAtom, 0], R1[iAtom, 1], R1[iAtom, 2])
#                sys.stdout.flush()

def getEngyFors(params):
    
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float64)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float64)

    tfCoord = tf.placeholder(tf.float64, shape=(None,3))
    tfLattice = tf.placeholder(tf.float64, shape=(3,3))
    
    tfEs,tfFs = tff.tf_getEF(tfCoord,tfLattice,params)
    tfEp = (tfEs - tfEngyB)/tfEngyA
    tfFp = tfFs/tfEngyA
    
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    with tf.Session() as sess:
        with open(params["inputData"], 'r') as mmtFile:
            nAtoms, lattice, R= getRmmt(mmtFile)
        feedDict={
                tfCoord:R,
                tfLattice: lattice
                }
        sess.run(tf.global_variables_initializer())
        if params["restart"]:
            saver.restore(sess, str(params['logDir'])+"/tf.chpt")
        Ep, Fp = sess.run((tfEp, tfFp), feed_dict=feedDict)
    return Ep, Fp


def initialize(params):
    import os
#    defaultParams={
#        "chunkSize": 0,
#        "epoch": 5000,
#        "restart": True,
#        "inputData": "MOVEMENT.train.first100",
#        "featFile": "feat",
#        "engyFile": "engy",
#        "logDir": "log",
#        "iGPU": 0,
#        "dcut": 6.2,
#        "learningRate": 0.0001,
#        "n2bBasis": 100,
#        "n3bBasis": 10,
#        "nL1Nodes": 300,
#        "nL2Nodes": 500
#        }
#    
#    for param in params:
#        defaultParams[param] = params[param]
#        
#    params = defaultParams
    
    file = open(str(params["inputData"]), 'r')
    nAtoms, iIter, lattice, R, f, v, e = getData(file)
    feat = getFeats(R, lattice, float(params['dcut']), params['n2bBasis'],params['n3bBasis'])
    engy = e.reshape([-1,1])
    file.close()
    
    featScalerA,featScalerB,engyScalerA,engyScalerB = getFeatEngyScaler(feat,engy)
    
    params['featScalerA'],params['featScalerB'],params['engyScalerA'],params['engyScalerB'] = \
    getFeatEngyScaler(feat,engy)
        
    if not os.path.exists(str(params['logDir'])):
        os.mkdir(str(params['logDir']))
            
    paramFile = str(params['logDir'])+"/params"
    np.savez(paramFile,**params)
    return params

def outputFeatures(engyFile, featFile, inputData, params):
    import os
    import pandas as pd
    
    if os.path.exists(featFile):
        os.remove(featFile)
    if os.path.exists(engyFile):
        os.remove(engyFile)
        
    tfR = tf.placeholder(tf.float64, shape=(None,3))
    tfL = tf.placeholder(tf.float64, shape=(3,3))
    
    nCase = 0
    with open(inputData, 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    sess = tf.Session()
    with open(inputData, 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = getData(datafile)
            feedDict = {tfR: R, tfL:lattice}
            feat = sess.run(
                    tff.tf_getFeatsFromR(tfR, tfL, float(params['dcut']), params['n2bBasis'],params['n3bBasis']),
                    feed_dict=feedDict)
            engy = e.reshape([-1,1])
            pd.DataFrame(feat).to_csv(featFile, mode='a', header=False, index=False)
            pd.DataFrame(engy).to_csv(engyFile, mode='a', header=False, index=False)

def trainEF(params):
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float64)
    tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float64)
    
    # Tensorflow placeholders
    tfCoord = tf.placeholder(tf.float64, shape=(None,3))
    tfLattice = tf.placeholder(tf.float64, shape=(3,3))
    tfEngy = tf.placeholder(tf.float64, shape=(None))
    tfFors = tf.placeholder(tf.float64, shape=(None,3))
    tfLearningRate = tf.placeholder(tf.float64)
    
    
    tfEs,tfFs = tff.tf_getEF(tfCoord,tfLattice,params)
    
    tfEp = (tfEs - tfEngyB)/tfEngyA
    tfFp = tfFs/tfEngyA
    
    tfLoss = tf.reduce_mean(tf.squared_difference(tfEs, tfEngy)) + \
             float(params['feRatio']) * tf.reduce_mean(tf.squared_difference(tfFs, tfFors))
             
    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
            tfOptimizer = tf.train.AdamOptimizer(tfLearningRate).minimize(tfLoss)
             
    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    def getError(mCase, tfileName):
        n = 0
        Ese = 0
        Eae = 0
        EseTot = 0
        Fse1 = 0
        Fse2 = 0
        with open(tfileName, "r") as tfile:
            for jCase in range(mCase):
                nAtoms, iIter, lattice, R, f, v, e = getData(tfile)
                engy = e.reshape([-1,1])
                feedDict={
                        tfCoord:R,
                        tfLattice: lattice,
                        }
                (Ep,Fp) = sess.run((tfEp,tfFp),feed_dict=feedDict)
                EseTot += (np.sum(Ep) - np.sum(engy))**2
                Ese += np.sum((Ep - engy)**2)
                Eae += np.sum(np.abs(Ep-engy))
                
                Fse1 += np.sum((np.sqrt(np.sum(f**2,1)) - np.sqrt(np.sum(Fp**2,1)))**2)
                Fse2 += np.sum((f-Fp)**2)
                
                n += len(engy)
        ErmseTot = np.sqrt(EseTot/mCase)
        Ermse = np.sqrt(Ese/n)
        Emae = Eae/n
        Frmse1 = np.sqrt(Fse1/n)
        Frmse2 = np.sqrt(Fse2/(3*n))
        print("Total Ermse: ", ErmseTot)
        print("Total Ermse per atom:", ErmseTot/nAtoms)
        print("Ermse: ", Ermse)
        print("Emae : ", Emae)
        print("Frmse (magnitude): ", Frmse1)
        print("Frmse (component): ", Frmse2)
        sys.stdout.flush()
    
    if params["restart"]:
        saver.restore(sess, str(params['logDir'])+"/tf.chpt")
        print("Model restored")
    
    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1
    vCase = 0
    if (params["validate"] >= 0) & (params["validationSet"] != ""):
        with open(str(params["validationSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    vCase += 1
            
    tCase = 0
    if (params["test"] >= 0) & (params["testSet"] != ""):
        with open(str(params["testSet"]), 'r') as datafile:
            for line in datafile:
                if "Iteration" in line:
                    tCase += 1

    for iEpoch in range(params["epoch"]):
        file = open(str(params["inputData"]), 'r')
        for iCase in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = getData(file)
            engy = e.reshape([-1,1])
            feedDict={
                    tfCoord:R,
                    tfLattice: lattice,
                    tfEngy: engy*params['engyScalerA']+params['engyScalerB'], 
                    tfFors: f*params['engyScalerA'],
                    tfLearningRate: float(params['learningRate']),
                    }
            sess.run(tfOptimizer,feed_dict=feedDict)
            
            (Ei,Fi) = sess.run((tfEp,tfFp),feed_dict=feedDict)
            Ermse = np.sqrt(np.mean((Ei-engy)**2))
            Frmse = np.sqrt(np.mean((Fi-f)**2))
            print(iEpoch, iCase, "Ermse:", Ermse)
            print(iEpoch, iCase, "Frmse:", Frmse)
            sys.stdout.flush()
    
        file.close()
        
        (Ei,Fi) = sess.run((tfEp,tfFp),feed_dict=feedDict)
        Ermse = np.sqrt(np.mean((Ei-engy)**2))
        Frmse = np.sqrt(np.mean((Fi-f)**2))
        print(iEpoch, "Ermse:", Ermse)
        print(iEpoch, "Frmse:", Frmse)
        sys.stdout.flush()
        
        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(vCase, str(params['validationSet']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print("Epoch", iEpoch)
                # print(str(iEpoch)+"th epoch")
                getError(tCase, str(params['testSet']))
        sys.stdout.flush()
            
    if params["validate"] == 0:
        getError(vCase, str(params['validationSet']))
    if params["test"] == 0:
        getError(tCase, str(params['testSet']))
        
    save_path = saver.save(sess, str(params['logDir'])+"/tf.chpt")
    return save_path