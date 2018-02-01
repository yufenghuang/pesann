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
    