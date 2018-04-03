#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:45:42 2018

@author: yufeng
"""

import tensorflow as tf
import numpy as np

def tf_getNb(tf_R, tf_lattice, dcut):
    # getNb: get neighbors
    # Inputs:
    #   tf_R [nAtoms x 3]:  the fractional coordinates of all atoms inside a unit cell
    #   tf_lattice [3x3]:   the lattice matrix that defines the unit cell
    #                       tf_lattice is defined by a set of column vectors [C1, C2, C3] \
    #                       such that the cartesian coordinates of an atom are given by \
    #                       X = x * C1 + y * C2 + z * C3
    #   tf_dcut [1]:        the cutoff radius
    #
    # Outputs:
    #   tf_idxNb [nAtoms x maxNb]:  A adjacency matrix that the list in row i gives 
    #                               the indices of all the atoms within dcut of atom i+1.
    #                               Note, row 0 as in Tensorflow or Numpy corresponds to atom 1
    #   tf_RNb [nAtoms x maxNb x 3]: 
    
#    tf_dcut = tf.constant(dcut, dtype=tf.float32)
    
    tf_Rd = tf.expand_dims(tf_R,0) - tf.expand_dims(tf_R,1)
    
    tf_RdShape = tf.shape(tf_Rd, out_type=tf.int64)
    
    tf_RdMaskPos = tf_Rd > 0.5
    tf_RdMaskNeg = tf_Rd < -0.5
    
    tf_Rd = tf.where(tf_RdMaskPos, \
                     tf.scatter_nd(tf.where(tf_RdMaskPos), tf.boolean_mask(tf_Rd, tf_RdMaskPos)-1, tf_RdShape), \
                     tf_Rd)
    tf_Rd = tf.where(tf_RdMaskNeg, \
                     tf.scatter_nd(tf.where(tf_RdMaskNeg), tf.boolean_mask(tf_Rd, tf_RdMaskNeg)+1, tf_RdShape), \
                     tf_Rd)
    
    tf_Rd = tf.reshape(tf.tensordot(tf_Rd, tf.transpose(tf_lattice),1), tf_RdShape)
    
    tf_dcutMask = tf.reduce_sum(tf_Rd**2, axis=2) < tf.reshape(dcut,[1])**2
    tf_Rd = tf.scatter_nd(tf.where(tf_dcutMask), tf.boolean_mask(tf_Rd, tf_dcutMask), tf_RdShape)
    
    tf_idxMask = tf.reduce_sum(tf_Rd**2, axis=2) > 0
    tf_numNb = tf.reduce_sum(tf.cast(tf_idxMask, tf.float32),axis=1)
    tf_maxNb = tf.cast(tf.reduce_max(tf_numNb),tf.int64)
    
    tf_idx = tf.transpose(tf.where(tf_idxMask))
    tf_iidx = tf_idx[0]
    tf_jidx = tf_idx[1]
#    tf_jidx2 = tf.cast(tf.concat([tf.range(tf_numNb[i]) for i in range(nAtoms)], axis=0),tf.int64)

    tfi0 = tf.constant(0,dtype=tf.int64)
    c = lambda i,x,y: i<tf.shape(x,out_type=tf.int64)[0]
    b = lambda i,x,y: [i+1,x,tf.concat([y,tf.range(x[i])],axis=0)]
    
    [o1,o2,o3] = tf.while_loop(c,b,\
                                [tfi0,tf.cast(tf_numNb,tf.int64),tf.zeros([1],dtype=tf.int64)],\
                                shape_invariants=[tfi0.get_shape(),tf_numNb.get_shape(),tf.TensorShape([None])])
    tf_jidx2 = o3[1:]

    tf_idx = tf.transpose(tf.stack([tf_iidx,tf_jidx2]))
    
    tf_idxNb = tf.scatter_nd(tf_idx, tf_jidx+1, [tf_RdShape[0], tf_maxNb])
    tf_RNb = tf.scatter_nd(tf_idx, tf.boolean_mask(tf_Rd, tf_idxMask), [tf_RdShape[0], tf_maxNb,3])

    return tf_idxNb,tf_RNb,tf_maxNb,tf_RdShape[0]

def tf_getStruct(tfCoord):
    tfRi = tf.sqrt(tf.reduce_sum(tfCoord**2,axis=2))
    tfDc = tf.sqrt(tf.reduce_sum((tf.expand_dims(tfCoord,2)-tf.expand_dims(tfCoord,1))**2,axis=3))
    idxRi = tf.where(tf.greater(tfRi, tf.constant(0.000000,dtype=tf.float32)))
    tfDc1 = tf.boolean_mask(tfDc, tf.greater(tfRi, tf.constant(0.000000,dtype=tf.float32)))
    tfDc2 = tf.scatter_nd(idxRi, tfDc1, tf.shape(tfDc,out_type=tf.int64))
    tfDc3 = tf.boolean_mask(tf.transpose(tfDc2, [0,2,1]),tf.greater(tfRi, tf.constant(0.000000,dtype=tf.float32)))
    tfDc4 = tf.scatter_nd(idxRi, tfDc3, tf.shape(tfDc,out_type=tf.int64))
    
    tfRi_masked = tf.boolean_mask(tfRi, tf.greater(tfRi, tf.constant(0.000000,dtype=tf.float32)))
    tfCoord_masked = tf.boolean_mask(tfCoord, tf.greater(tfRi, tf.constant(0.000000,dtype=tf.float32)))
    tfRhat1 = tfCoord_masked/tf.expand_dims(tfRi_masked,1)
    tfRhat2 = tf.scatter_nd(idxRi, tfRhat1, tf.shape(tfCoord,out_type=tf.int64))
    return tfRhat2, tfRi, tfDc4

def tf_getCos(tf_X,tf_nBasis):
    tf_pi = tf.constant(np.pi, tf.float32)
#    tf_X = tf.placeholder(tf.float32,[None])
#    tf_nBasis = tf.placeholder(tf.int32)
    tf_Y = tf.expand_dims(tf_X,1) - tf.linspace(tf.constant(-1.,dtype=tf.float32),
                          tf.constant(1., dtype=tf.float32),tf_nBasis)
    tf_h = tf.cast(2/(tf_nBasis-1),tf.float32)
    tf_zeroMask = tf.equal(tf_Y, 0.)
    tf_Y = tf.reshape(tf.where(tf.abs(tf_Y) < tf_h, tf_Y, tf.zeros_like(tf_Y)),[-1,tf_nBasis])
    tf_Ynot0 = tf.not_equal(tf_Y, 0.)
    tf_Y = tf.scatter_nd(tf.where(tf_Ynot0), \
                         tf.cos(tf.boolean_mask(tf_Y, tf_Ynot0)/tf_h*tf_pi)/2+0.5, \
                         tf.shape(tf_Y, out_type=tf.int64))
    tf_Y = tf.where(tf_zeroMask, tf.ones_like(tf_Y), tf_Y)
    tf_Y = tf.where(tf.abs(tf_X)>1., tf.zeros_like(tf_Y), tf_Y)
    return tf_Y

def tf_getdCos(tf_X,tf_nBasis):
    tf_pi = tf.constant(np.pi, tf.float32)
#    tf_Y = tf.expand_dims(tf_X,1) - tf.linspace(-1.,1.,tf_nBasis)
    tf_Y = tf.expand_dims(tf_X,1) - tf.linspace(tf.constant(-1.,dtype=tf.float32),
                          tf.constant(1., dtype=tf.float32),tf_nBasis)

    tf_h = tf.cast(2/(tf_nBasis-1),tf.float32)
    tf_Y = tf.reshape(tf.where(tf.abs(tf_Y) < tf_h, tf_Y, tf.zeros_like(tf_Y)),[-1,tf_nBasis])
    tf_Ynot0 = tf.not_equal(tf_Y, 0.)
    tf_Y = tf.scatter_nd(tf.where(tf_Ynot0), \
                         -tf.sin(tf.boolean_mask(tf_Y, tf_Ynot0)/tf_h*tf_pi)*0.5*tf_pi/tf_h, \
                         tf.shape(tf_Y, out_type=tf.int64))
    tf_Y = tf.where(tf.abs(tf_X)>1, tf.zeros_like(tf_Y), tf_Y)
    return tf_Y


def tf_getCos2(tf_X,tf_nBasis):
    # Cosine basis functions with no discontinuing basis functions at the boundary
    
    tf_pi = tf.constant(np.pi, tf.float32)
    tf_h = tf.cast(2/(tf_nBasis),tf.float32)
    
    tf_Y = tf.expand_dims(tf_X,1) - tf.linspace(tf.constant(-1.,dtype=tf.float32),
                          tf.constant(1., dtype=tf.float32)-tf_h,tf_nBasis)
        
    tf_zeroMask = tf.equal(tf_Y, 0.)
    tf_Y = tf.reshape(tf.where(tf.abs(tf_Y) < tf_h, tf_Y, tf.zeros_like(tf_Y)),[-1,tf_nBasis])
    tf_Ynot0 = tf.not_equal(tf_Y, 0.)
    tf_Y = tf.scatter_nd(tf.where(tf_Ynot0), \
                         tf.cos(tf.boolean_mask(tf_Y, tf_Ynot0)/tf_h*tf_pi)/2+0.5, \
                         tf.shape(tf_Y, out_type=tf.int64))
    tf_Y = tf.where(tf_zeroMask, tf.ones_like(tf_Y), tf_Y)
    
    tf_Y = tf.where(tf_X>1., tf.zeros_like(tf_Y), tf_Y)
    tf_Y = tf.where(tf_X<(-1.-tf_h), tf.zeros_like(tf_Y), tf_Y)
#    tf_Y = tf.where(tf.abs(tf_X)>1, tf.zeros_like(tf_Y), tf_Y)
    return tf_Y

def tf_getdCos2(tf_X,tf_nBasis):
    # Derivative of the sosine basis functions with no discontinuing basis functions at the boundary
    tf_pi = tf.constant(np.pi, tf.float32)
    
    tf_h = tf.cast(2/(tf_nBasis),tf.float32)
    tf_Y = tf.expand_dims(tf_X,1) - tf.linspace(tf.constant(-1.,dtype=tf.float32),
                          tf.constant(1., dtype=tf.float32)-tf_h,tf_nBasis)

    tf_Y = tf.reshape(tf.where(tf.abs(tf_Y) < tf_h, tf_Y, tf.zeros_like(tf_Y)),[-1,tf_nBasis])
    tf_Ynot0 = tf.not_equal(tf_Y, 0.)
    tf_Y = tf.scatter_nd(tf.where(tf_Ynot0), \
                         -tf.sin(tf.boolean_mask(tf_Y, tf_Ynot0)/tf_h*tf_pi)*0.5*tf_pi/tf_h, \
                         tf.shape(tf_Y, out_type=tf.int64))
#    tf_Y = tf.where(tf.abs(tf_X)>1, tf.zeros_like(tf_Y), tf_Y)
    tf_Y = tf.where(tf_X>1., tf.zeros_like(tf_Y), tf_Y)
    tf_Y = tf.where(tf_X<(-1.-tf_h), tf.zeros_like(tf_Y), tf_Y)
    return tf_Y


def tf_engyFromFeats(tfFeats, nFeat, nL1, nL2):
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        W = tf.get_variable("weights", shape=[nFeat,nL1], dtype=tf.float32, 
              initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases",shape=[nL1], dtype=tf.float32,
             initializer=tf.zeros_initializer())
        L1out = tf.nn.sigmoid(tf.matmul(tfFeats, W)+B)
        tf.add_to_collection("saved_params", W)
        tf.add_to_collection("saved_params", B)
        
    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        W = tf.get_variable("weights", shape=[nL1,nL2], dtype=tf.float32, 
              initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases",shape=[nL2], dtype=tf.float32,
             initializer=tf.zeros_initializer())
        L2out = tf.nn.sigmoid(tf.matmul(L1out, W)+B)
        tf.add_to_collection("saved_params", W)
        tf.add_to_collection("saved_params", B)
        
    with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
        W = tf.get_variable("weights", shape=[nL2,1], dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases",shape=[1], dtype=tf.float32,
             initializer=tf.zeros_initializer())
        tf.add_to_collection("saved_params", W)
        tf.add_to_collection("saved_params", B)

        L3out = tf.matmul(L2out, W)+B
    return L3out

def tf_get_dEldXi(tfFeats, nFeat, nL1, nL2):
    with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable("weights", shape=[nFeat,nL1],dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases",shape=[nL1],dtype=tf.float32,
             initializer=tf.zeros_initializer())
        L1out = tf.nn.sigmoid(tf.matmul(tfFeats, W1)+B)
        
    with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
        W2 = tf.get_variable("weights", shape=[nL1,nL2],dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases",shape=[nL2],dtype=tf.float32,
             initializer=tf.zeros_initializer())
        L2out = tf.nn.sigmoid(tf.matmul(L1out, W2)+B)
        
    with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
        W3 = tf.get_variable("weights", shape=[nL2,1], dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        
    w_j = tf.reduce_sum(tf.expand_dims(L2out*(1-L2out),1) * \
                       tf.expand_dims(W2,0) * \
                       tf.expand_dims(tf.transpose(W3),0), axis=2)
    dEldXi = tf.reduce_sum(tf.expand_dims(L1out*(1-L1out)*w_j,1) * tf.expand_dims(W1,0),2)
    return dEldXi

def tf_getEF(tfCoord, tfLattice,params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    numFeat = params['n2bBasis'] + params['n3bBasis']**3
        
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tf_getNb(tfCoord,tfLattice,float(params['dcut']))
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)
    
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tf_getCos2(tf.boolean_mask(tfDc,tfDc>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    
    tfdXi, tfdXin = tf_get_dXidRl(tfGR2,tfGR2d,tfGR3,tfGR3d,tfGD3,tfRhat*3/params['dcut'])
    tfdXi =  tf.expand_dims(tfFeatA,2) * tfdXi 
    tfdXin =  tf.expand_dims(tfFeatA,2) * tfdXin
    
    tfFeats = tfFeatA*tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
    tfEs = tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    
    dEldXi = tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi,2)*tfdXi,axis=1)
    
    dENldXi=tf.gather_nd(dEldXi,tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb,0))-1),1))
    dEnldXin=tf.scatter_nd(tf.where(tf.greater(tfIdxNb,0)), dENldXi, [tfNAtoms,tfMaxNb,numFeat])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*tfdXin,axis=[1,2])
    
    tfFs = Fln + Fll 

    return tfEs, tfFs

def tf_getEF2(tfCoord, tfLattice,params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    numFeat = params['n2bBasis'] + params['n3bBasis']**3
        
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tf_getNb(tfCoord,tfLattice,float(params['dcut']))
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)
    
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    
    tfdXi, tfdXin = tf_get_dXidRl(tfGR2,tfGR2d,tfGR3,tfGR3d,tfGD3,tfRhat)
    tfdXi =  tf.expand_dims(tfFeatA,2) * tfdXi 
    tfdXin =  tf.expand_dims(tfFeatA,2) * tfdXin
    
    tfFeats = tfFeatA*tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
    tfEs = tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    
    dEldXi = tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi,2)*tfdXi,axis=1)
    
    dENldXi=tf.gather_nd(dEldXi,tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb,0))-1),1))
    dEnldXin=tf.scatter_nd(tf.where(tf.greater(tfIdxNb,0)), dENldXi, [tfNAtoms,tfMaxNb,numFeat])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*tfdXin,axis=[1,2])
    
    tfFs = Fln + Fll 

    return tfEs, tfFs, Fll, Fln

def tf_getEF3(tfCoord, tfLattice,params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    numFeat = params['n2bBasis'] + params['n3bBasis']**3
        
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tf_getNb(tfCoord,tfLattice,float(params['dcut']))
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)
    
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi>0),\
                           tf_getdCos(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                           [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tf_getCos(tf.boolean_mask(tfDc,tfDc>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
    
    tfdXi, tfdXin = tf_get_dXidRl(tfGR2,tfGR2d,tfGR3,tfGR3d,tfGD3,tfRhat)
    tfdXi =  tf.expand_dims(tfFeatA,2) * tfdXi 
    tfdXin =  tf.expand_dims(tfFeatA,2) * tfdXin
    
    tfFeats = tfFeatA*tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
    tfEs = tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    
    dEldXi = tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi,2)*tfdXi,axis=1)
    
    dENldXi=tf.gather_nd(dEldXi,tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb,0))-1),1))
    dEnldXin=tf.scatter_nd(tf.where(tf.greater(tfIdxNb,0)), dENldXi, [tfNAtoms,tfMaxNb,numFeat])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*tfdXin,axis=[1,2])
    
    tfFs = Fln + Fll 

    return tfEs, tfFs, tfFeats, tfdXi, dEldXi


def tf_getE(tfCoord, tfLattice,params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    numFeat = params['n2bBasis'] + params['n3bBasis']**3
        
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tf_getNb(tfCoord,tfLattice,float(params['dcut']))
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)
    
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n2bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb,params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tf_getCos2(tf.boolean_mask(tfDc,tfDc>0)*3/float(params['dcut'])-2,params['n3bBasis']),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,params['n3bBasis']])
        
    tfFeats = tfFeatA*tf_getFeats(tfGR2,tfGR3,tfGD3)+tfFeatB
    tfEs = tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])

    return tfEs


def tf_get_dXidRl(tf_GR2, tf_GR2d, tf_GR3, tf_GR3d, tf_GD3, tf_Rh):
    
    tf_Shape = tf.shape(tf_GR3,out_type=tf.int64)
    tf_maxNb = tf.reshape(tf_Shape[1],[1])
    tf_nBasis3b = tf.reshape(tf_Shape[2],[1])
    
    tfX1d = tf.reduce_sum(tf.expand_dims(tf_GR2d,3) * tf.expand_dims(-tf_Rh,2),1) # dX2/dRl
    
    tfX2da = tf.reduce_sum(tf.expand_dims(tf_GD3,3) * tf.expand_dims(tf.expand_dims(tf_GR3,2),4),1)
    tfX2db = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf_GR3d,3),4) * tf.expand_dims(tfX2da,2),5) * \
                           tf.expand_dims(tf.expand_dims(tf.expand_dims(-tf_Rh,2),3),4), 1)
    tfX2d = tfX2db + tf.transpose(tfX2db,[0,2,1,3,4]) # dX3/dRl
    
    tfX1dn = tf.expand_dims(tf_GR2d,3) * tf.expand_dims(-tf_Rh,2) # dX2/dRnl
    
    tfX2dn_a = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GR3,1),3) * tf.expand_dims(tf_GD3,4),2)
    tfX2dn_b = tf.expand_dims(tf.expand_dims(tfX2dn_a,2),5) * \
               tf.expand_dims(tf.expand_dims(tf.expand_dims(-tf_Rh, 2),3),4) * \
               tf.expand_dims(tf.expand_dims(tf.expand_dims(tf_GR3d,3),4),5)
    tfX2dn1 = tfX2dn_b + tf.transpose(tfX2dn_b, [0,1,3,2,4,5])
    
    tfX2dn_c = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GD3,4),5) * \
               tf.expand_dims(tf.expand_dims(tf.expand_dims(tf_GR3d,1),3),5) * \
               tf.expand_dims(tf.expand_dims(tf.expand_dims(-tf_Rh,1),3),4),2)
    tfX2dn_d = tf.expand_dims(tfX2dn_c,2) * tf.expand_dims(tf.expand_dims(tf.expand_dims(tf_GR3,3),4),5)
    tfX2dn2 = tfX2dn_d + tf.transpose(tfX2dn_d,[0,1,3,2,4,5])
    
    tfShapeTemp = tf.concat([[-1],tf_maxNb,tf_nBasis3b**3,[3]],axis=0)
    tfX2dn = tf.reshape(tfX2dn1 + tfX2dn2, tfShapeTemp) # dX3/dRnl    
    
    tfShapeTemp = tf.concat([[-1],tf_nBasis3b**3,[3]],axis=0)
    tfXd = tf.concat([tfX1d, tf.reshape(tfX2d,tfShapeTemp)],axis=1)
    tfXdn = tf.concat([tfX1dn, tfX2dn],axis=2)

    return tfXd,tfXdn

def tf_getFeats(tf_GR2, tf_GR3, tf_GD3):
    tf_n3bBasis = tf.shape(tf_GR3)[2]
    tf_yR = tf.reduce_sum(tf_GR2,axis=1)
    tf_yD = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GR3,1),4) * tf.expand_dims(tf_GD3,3),2)
    tf_yD = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GR3,3),4) * tf.expand_dims(tf_yD,2),1)
    tf_yD = tf.reshape(tf_yD,[-1,tf_n3bBasis**3])
    return tf.concat([tf_yR, tf_yD],axis=1)

def tf_getFeatsFromR(tfCoord, tfLattice, dcut,n2bBasis, n3bBasis):
    tfIdxNb, tfRNb,tfMaxNb, tfNAtoms= tf_getNb(tfCoord,tfLattice,dcut)
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)
    tfGR2 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n2bBasis),\
                          [tfNAtoms,tfMaxNb,n2bBasis])
    tfGR3 = tf.scatter_nd(tf.where(tfRi>0),\
                          tf_getCos2(tf.boolean_mask(tfRi,tfRi>0)*3/dcut-2,n3bBasis),\
                          [tfNAtoms,tfMaxNb,n3bBasis])
    tfGD3 = tf.scatter_nd(tf.where(tfDc>0),\
                          tf_getCos2(tf.boolean_mask(tfDc,tfDc>0)*3/dcut-2,n3bBasis),\
                          [tfNAtoms,tfMaxNb, tfMaxNb,n3bBasis])
    tfFeats = tf_getFeats(tfGR2,tfGR3,tfGD3)
    return tfFeats

def tf_getFc(Ri, Rc):
    return 0.5*(tf.cos(np.pi*Ri/Rc)+1)

def tf_getdFc(Ri, Rc):
    return -0.5*np.pi/Rc*tf.sin(np.pi*Ri/Rc)

def tf_getEa(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0*Rci**12/tf_getFc(Rci,Rc)
    return alpha*tf_getFc(Ri, Rc)/Ri**12

def tf_getdEa(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0*Rci**12/tf_getFc(Rci,Rc)
    return -12*alpha*tf_getFc(Rci, Rc)/Ri**13 + alpha*tf_getdFc(Ri, Rc)/Ri**12

def tf_getEb(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0 * Rci/tf_getFc(Rci, Rc)
    return alpha*tf_getFc(Ri, Rc)/Ri

def tf_getdEb(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0 * Rci/tf_getFc(Rci,Rc)
    return -alpha*tf_getFc(Ri, Rc)/Ri**2 + alpha*tf_getdFc(Ri, Rc)/Ri

def tf_getEc(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0 * tf.exp(1.)/tf_getFc(Rci, Rc)
    return alpha * tf.exp(-Ri/Rci)*tf_getFc(Ri, Rc)

def tf_getdEc(Ri, Rc, Rci):
    E0 = 0.1
    alpha = E0 * tf.exp(1.)/tf_getFc(Rci, Rc)
    return -alpha/Rci*tf.exp(-Ri/Rci)*tf_getFc(Ri, Rc) + alpha * tf.exp(-Ri/Rci)*tf_getdFc(Ri, Rc)

def tf_getVi(tfRi, Rc, Rci, params):
    if params["repulsion"] == "None":
        return tf.reduce_sum(tfRi,axis=1)*0
    elif params["repulsion"] == "1/R12":
        Ea = tf.scatter_nd(tf.where(tfRi > 0), tf_getEa(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                           tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(Ea, axis=1)

    elif params["repulsion"] == "1/R":
        Eb = tf.scatter_nd(tf.where(tfRi > 0), tf_getEb(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                           tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(Eb, axis=1)
    elif params["repulsion"] == "exp(-R)":
        Ec = tf.scatter_nd(tf.where(tfRi > 0), tf_getEc(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                           tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(Ec, axis=1)
    else:
        print("Unknown repulsion term. Ignoring repulsion...")
        return tf.reduce_sum(tfRi,axis=1)*0

def tf_getdVi(tfRi, tfRhat, Rc, Rci, params):
    if params["repulsion"] == "None":
        return tf.reduce_sum(tfRhat, axis=1)*0
    elif params["repulsion"] == "1/R12":
        dEa = tf.scatter_nd(tf.where(tfRi > 0), tf_getdEa(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                             tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(tf.expand_dims(dEa, 2) * (-tfRhat), axis=1)
    elif params["repulsion"] == "1/R":
        dEb =  tf.scatter_nd(tf.where(tfRi > 0), tf_getdEb(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                             tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(tf.expand_dims(dEb, 2) * (-tfRhat), axis=1)
    elif params["repulsion"] == "exp(-R)":
        dEc =  tf.scatter_nd(tf.where(tfRi > 0), tf_getdEc(tf.boolean_mask(tfRi, tfRi > 0), Rc, Rci),
                             tf.shape(tfRi, out_type=tf.int64))
        return tf.reduce_sum(tf.expand_dims(dEc, 2) * (-tfRhat), axis=1)
    else:
        print("Unknown repulsion term. Ignoring repulsion...")
        return tf.reduce_sum(tfRhat, axis=1)*0

def tf_getEF_repulsion(tfCoord, tfLattice, params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)

    numFeat = params['n2bBasis'] + params['n3bBasis'] ** 3

    tfIdxNb, tfRNb, tfMaxNb, tfNAtoms = tf_getNb(tfCoord, tfLattice, float(params['dcut']))
    tfRhat, tfRi, tfDc = tf_getStruct(tfRNb)

    tfGR2 = tf.scatter_nd(tf.where(tfRi > 0),
                          tf_getCos2(tf.boolean_mask(tfRi, tfRi > 0) * 3 / float(params['dcut']) - 2,
                                     params['n2bBasis']),
                          [tfNAtoms, tfMaxNb, params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi > 0),
                           tf_getdCos2(tf.boolean_mask(tfRi, tfRi > 0) * 3 / float(params['dcut']) - 2,
                                       params['n2bBasis']),
                           [tfNAtoms, tfMaxNb, params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi > 0),
                          tf_getCos2(tf.boolean_mask(tfRi, tfRi > 0) * 3 / float(params['dcut']) - 2,
                                     params['n3bBasis']),
                          [tfNAtoms, tfMaxNb, params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi > 0),
                           tf_getdCos2(tf.boolean_mask(tfRi, tfRi > 0) * 3 / float(params['dcut']) - 2,
                                       params['n3bBasis']),
                           [tfNAtoms, tfMaxNb, params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc > 0),
                          tf_getCos2(tf.boolean_mask(tfDc, tfDc > 0) * 3 / float(params['dcut']) - 2,
                                     params['n3bBasis']),
                          [tfNAtoms, tfMaxNb, tfMaxNb, params['n3bBasis']])

    tfdXi, tfdXin = tf_get_dXidRl(tfGR2, tfGR2d, tfGR3, tfGR3d, tfGD3, tfRhat * 3 / params['dcut'])
    tfdXi = tf.expand_dims(tfFeatA, 2) * tfdXi
    tfdXin = tf.expand_dims(tfFeatA, 2) * tfdXin

    tfFeats = tfFeatA * tf_getFeats(tfGR2, tfGR3, tfGD3) + tfFeatB
    tfEs = tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes']) + \
           tf.expand_dims(tf_getVi(tfRi, params['dcut'], params['dcut']/3, params), 1) * tfEngyA

    dEldXi = tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi, 2) * tfdXi, axis=1)

    dENldXi = tf.gather_nd(dEldXi,
                           tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb, 0)) - 1), 1))
    dEnldXin = tf.scatter_nd(tf.where(tf.greater(tfIdxNb, 0)), dENldXi, [tfNAtoms, tfMaxNb, numFeat])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin, 3) * tfdXin, axis=[1, 2])

    tfFs = Fln + Fll + 2 * tfEngyA * tf_getdVi(tfRi, tfRhat, params['dcut'], params['dcut']/3, params)

    return tfEs, tfFs
