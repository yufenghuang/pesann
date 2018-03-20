
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:19:55 2018

@author: hif10
"""

import numpy as np

def np_getNb(np_R, np_lattice, dcut):
    
    nAtoms = np_R.shape[0]

    Rd = np_R[None,:,:] - np_R[:,None,:]
    Rd[Rd>0.5] = Rd[Rd>0.5] - 1
    Rd[Rd<-0.5] = Rd[Rd<-0.5] + 1
    
    Rd = np.dot(Rd, np_lattice.T)
    
    dcutMask = np.sum(Rd**2,axis=2) < dcut**2
    Rd[~dcutMask] = 0
    
    idxMask = np.sum(Rd**2,axis=2)>0
    numNb = idxMask.sum(axis=1)
    iidx, jidx = np.where(idxMask)
    jidx2 = np.concatenate([np.arange(numNb[i]) for i in range(nAtoms)])
    
    maxNb = np.max(numNb)
    
    idxNb = np.zeros((nAtoms,maxNb)).astype(int)
    idxNb[(iidx,jidx2)] = jidx+1
    
    coord = np.zeros((nAtoms,maxNb,3))
    coord[(iidx,jidx2)] = Rd[idxMask]

    return idxNb, coord, maxNb, nAtoms

def getCos(x, numBasis):
    yLPP = np.linspace(-1,1,numBasis)
    h = yLPP[1:] - yLPP[:-1]
    
    xy =  x[:,np.newaxis] - yLPP
    zeroMask = (xy==0)
    xy[xy>np.concatenate((h, [0]))] = 0
    xy[xy<-np.concatenate(([0],h))] = 0
    
    xyR = xy[:,:-1]
    xyL = xy[:,1:]
    
    (xy[:,:-1])[xyR>0] = np.cos((xyR/h)[xyR>0]*np.pi)/2+0.5
    (xy[:,1:])[xyL<0] = np.cos((xyL/h)[xyL<0]*np.pi)/2+0.5
    xy[zeroMask]=1
    return xy

def getdCos(x, numBasis):
    yLPP = np.linspace(-1,1,numBasis)
    h = yLPP[1:] - yLPP[:-1]
    
    xy =  x[:,np.newaxis] - yLPP
    xy[xy>np.concatenate((h, [0]))] = 0
    xy[xy<-np.concatenate(([0],h))] = 0
    
    xyR = xy[:,:-1]
    xyL = xy[:,1:]
    
    (xy[:,:-1])[xyR>0] = np.sin((xyR/h)[xyR>0]*np.pi)
    (xy[:,1:])[xyL<0] = np.sin((xyL/h)[xyL<0]*np.pi)
    (xy[:,:-1])[xyR>0] = (0.5*np.pi*xy[:,:-1]/h)[xyR>0]
    (xy[:,1:])[xyL<0] = (0.5*np.pi*xy[:,1:]/h)[xyL<0]
    return -xy

def getStruct(coord):
    Ri = np.sqrt(np.sum(coord**2,axis=2))
    Dc = np.sqrt(np.sum((coord[:,:,np.newaxis,:]-coord[:,np.newaxis,:,:])**2,axis=3))
    Dc[Ri==0] = 0
    Dc.transpose([0,2,1])[Ri==0] = 0
    Rhat = np.zeros_like(coord)
    Rhat[Ri>0] = coord[Ri>0]/Ri[Ri>0,np.newaxis]
    return Rhat, Ri, Dc

def tf_getFeats(tf_GR2, tf_GR3, tf_GD3):
    tf_n3bBasis = tf.shape(tf_GR3)[2]
    tf_yR = tf.reduce_sum(tf_GR2,axis=1)
    tf_yD = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GR3,1),4) * tf.expand_dims(tf_GD3,3),2)
    tf_yD = tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf_GR3,3),4) * tf.expand_dims(tf_yD,2),1)
    tf_yD = tf.reshape(tf_yD,[-1,tf_n3bBasis**3])
    return tf.concat([tf_yR, tf_yD],axis=1)


def np_getFeats(gR2, gR3, gD3):
    nb3 = gR3.shape[2]
    yR = gR2.sum(axis=1)
    yD = (gR3[:,None,:,:,None] * gD3[:,:,:,None,:]).sum(axis=2)
    yD = (gR3[:,:,:,None,None] * yD[:,:,None,:,:])

def getFeatures(Ri, Dc,numLPP_Rlpp,numLPP_D):
    yR_lpp = np.zeros((Ri.shape[0], Ri.shape[1], numLPP_Rlpp))
    yR_lpp[Ri>0] = getCos(Ri[Ri>0]*3/Rc-2, numLPP_Rlpp)
    yR = yR_lpp.sum(axis=1)

    yDR = np.zeros((Ri.shape[0], Ri.shape[1],numLPP_D))
    yDc = np.zeros((Dc.shape[0], Dc.shape[1], Dc.shape[2],numLPP_D))
    yDR[Ri>0] = getCos(Ri[Ri>0]*3/Rc-2, numLPP_D)
    yDc[Dc>0] = getCos(Dc[Dc>0]*3/Rc-2, numLPP_D)
    yD1 = (yDc[:,:,:,np.newaxis,:] * yDR[:,np.newaxis,:,:,np.newaxis]).sum(axis=2)
    yD2 = (yD1[:,:,np.newaxis,:,:] * yDR[:,:,:,np.newaxis,np.newaxis]).sum(axis=1)
    yDlpp = yD2.reshape([Ri.shape[0],-1])

    return yR, yDlpp

def getdFeatures(Ri, Dc, Rhat,numLPP_Rlpp,numLPP_D):
    dyR = np.zeros((Ri.shape[0], Ri.shape[1], numLPP_Rlpp))
    dyR[Ri>0] = getdCos(Ri[Ri>0]*3/Rc-2, numLPP_Rlpp)*3/Rc
    dyR = (dyR[:,:,:,np.newaxis]*Rhat[:,:,np.newaxis,:]).sum(axis=1)
    
    dyRn = np.zeros((Ri.shape[0], Ri.shape[1], numLPP_Rlpp))
    dyRn[Ri>0] = getdCos(Ri[Ri>0]*3/Rc-2, numLPP_Rlpp)*3/Rc
    dyRn = dyRn[:,:,:,np.newaxis] * Rhat[:,:,np.newaxis,:]

    yDR = np.zeros((Ri.shape[0], Ri.shape[1],numLPP_D))
    yDc = np.zeros((Dc.shape[0], Dc.shape[1], Dc.shape[2],numLPP_D))
    yDR[Ri>0] = getCos(Ri[Ri>0]*3/Rc-2, numLPP_D)
    yDc[Dc>0] = getCos(Dc[Dc>0]*3/Rc-2, numLPP_D)
    yD1 = (yDc[:,:,:,np.newaxis,:] * yDR[:,np.newaxis,:,:,np.newaxis]).sum(axis=2)
    
    dyD = np.zeros((Ri.shape[0], Ri.shape[1], numLPP_D))
    dyD[Ri>0] = getdCos(Ri[Ri>0]*3/Rc-2, numLPP_D)*3/Rc
    dyD_half = (Rhat[:,:,np.newaxis,np.newaxis,np.newaxis,:] * \
            yD1[:,:,np.newaxis,:,:,np.newaxis] * \
            dyD[:,:,:,np.newaxis,np.newaxis,np.newaxis]).sum(axis=1)
    dyDcos = dyD_half.transpose([0,2,1,3,4]) + dyD_half

    dyDn = np.zeros((Ri.shape[0], Ri.shape[1], numLPP_D))
    dyDn[Ri>0] = getdCos(Ri[Ri>0]*3/Rc-2, numLPP_D)*3/Rc
    
    dX2indRl = yD1.transpose([0,1,3,2])[:,:,np.newaxis,:,:] * dyDn[:,:,:,np.newaxis,np.newaxis]
    dX2indRl = dX2indRl + dX2indRl.transpose([0,1,3,2,4])
    dX2indRl = dX2indRl[:,:,:,:,:,np.newaxis] * Rhat[:,:,np.newaxis,np.newaxis,np.newaxis,:]
    
    dX2indRl2 = np.sum(yDc[:,:,:,:,np.newaxis,np.newaxis] * dyD[:,:,np.newaxis,np.newaxis,:,np.newaxis] * \
                Rhat[:,:,np.newaxis,np.newaxis,np.newaxis,:],axis=1)
#    dX2indRl2 = dX2indRl2[:,:,np.newaxis,:,:,:] * dyDn[:,:,:,np.newaxis,np.newaxis,np.newaxis] #found the mistake!!
    dX2indRl2 = dX2indRl2[:,:,np.newaxis,:,:,:] * yDR[:,:,:,np.newaxis,np.newaxis,np.newaxis]
    dX2indRl2 = dX2indRl2.transpose([0,1,3,2,4,5]) + dX2indRl2
    
    dyDn = (dX2indRl + dX2indRl2).reshape([Ri.shape[0],maxNb,numLPP_D**3,3])
    
    return -dyR, -dyDcos, -dyRn, -dyDn