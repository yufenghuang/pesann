import numpy as np
import py_func as pyf
import np_func as npf
import tensorflow as tf
import tf_func as tff

import pandas as pd

import os
import sys

import re

def fc(Rij, Rc):
    Rij_ = np.array(Rij)
    R_ = np.zeros(Rij_.shape)
    R_[Rij_<Rc] = 0.5*(np.cos(Rij_[Rij_<Rc]*np.pi/Rc)+1)
    return R_

def g1(Rij,Rs_,eta_, Rc):
    return np.exp(-eta_*(Rij-Rs_)**2)*fc(Rij, Rc)

def g2(Rij,Rik,Rjk,eta_,zeta_,ll_, Rc): ##ll: lambda
    cos = (Rij**2+Rik**2-Rjk**2)/(2*Rij*Rik)
    return 2.**(1-zeta_)*(1+ll_*cos)**zeta_ * np.exp(-eta_*(Rij**2+Rik**2+Rjk**2))* \
            fc(Rij, Rc)*fc(Rik, Rc)*fc(Rjk, Rc)

def getGaussianFeats(ll, eta, zeta, Rs, Rc, Ri, Di, Dj, Dc):
    feats = np.zeros((len(Ri), eta.size * Rs.size + ll.size * zeta.size * eta.size))
    for i in range(Rs.size):
        for j in range(eta.size):
            ifeat = i * eta.size + j
            G1 = np.zeros(Ri.shape)
            G1[Ri > 0.] = g1(Ri[Ri > 0.], Rs[i], eta[j], Rc)
            feats[:, ifeat] = G1.sum(axis=1)

    for i in range(ll.size):
        for j in range(eta.size):
            for k in range(zeta.size):
                ifeat = Rs.size * eta.size + i * eta.size * zeta.size + j * zeta.size + k
                G2 = np.zeros(Dc.shape)
                G2[Dc > 0.] = g2(Di[Dc > 0.], Dj[Dc > 0.], Dc[Dc > 0.], eta[j], zeta[k], ll[i], Rc)
                feats[:, ifeat] = G2.sum(axis=2).sum(axis=1)
    return feats

def specialTask01(engyFile, featFile, inputData, params):
    #   Generate features for the Gaussian symmetry functions
    #
    Rc = float(params["dcut"])
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size

    # featFile = str(params["featFile"])
    # engyFile = str(params["engyFile"])

    nCase = 0
    with open(inputData, 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    with open(inputData, 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
            idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params["dcut"]))
            Rhat, Ri, Dc = npf.getStruct(coord)
            Di = Ri[:,:,None] * np.ones(maxNb)
            Di[Dc == 0] = 0
            Dj = np.transpose(Di, [0,2,1])
            feat = getGaussianFeats(ll, eta, zeta, Rs, Rc, Ri, Di, Dj, Dc)
            engy = e.reshape([-1, 1])
            pd.DataFrame(feat).to_csv(featFile, mode='a', header=False, index=False)
            pd.DataFrame(engy).to_csv(engyFile, mode='a', header=False, index=False)
            
def specialTask02(params):
    Rc = float(params["dcut"])
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size
    numFeat = num2bFeat + num3bFeat

    tfFeat = tf.placeholder(tf.float32, shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float32, shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)

    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])

    tfLoss = tf.reduce_mean((tfEs - tfEngy) ** 2)

    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        tfOptimizer = tf.train.AdamOptimizer(tfLR).minimize(tfLoss)

    saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def getError(fFile, eFile):
        featDF = pd.read_csv(fFile, header=None, index_col=False).values
        engyDF = pd.read_csv(eFile, header=None, index_col=False).values
        feedDict2 = {tfFeat: featDF * params['featScalerA'] + params['featScalerB'],
                     tfEngy: engyDF * params['engyScalerA'] + params['engyScalerB']}
        Ep2 = sess.run(tfEs, feed_dict=feedDict2)
        Ep2 = (Ep2 - params['engyScalerB']) / params['engyScalerA']

        # print(Ep2[:,:])
        # print(engyDF[:,:])

        Ermse = np.sqrt(np.mean((Ep2.reshape(-1) - engyDF.reshape(-1)) ** 2))
        Emae = np.mean(np.abs(Ep2.reshape(-1) - engyDF.reshape(-1)))
        print("Ermse is: ", Ermse)
        print("Emae is : ", Emae)
        sys.stdout.flush()

    if params["restart"]:
        saver.restore(sess, str(params['logDir']) + "/tf.chpt")
        print("Model restored")

    if params['chunkSize'] == 0:
        dfFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False).values
        dfEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False).values

    for iEpoch in range(params['epoch']):
        if params['chunkSize'] > 0:
            pdFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False,
                                 chunksize=int(params['chunkSize']), iterator=True)
            pdEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False,
                                 chunksize=int(params['chunkSize']), iterator=True)

            for pdF in pdFeat:
                pdE = next(pdEngy)
                dfFeat = pdF.values
                dfEngy = pdE.values
                feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'],
                            tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'],
                            tfLR: params['learningRate']}

                sess.run(tfOptimizer, feed_dict=feedDict)
        #                print("running",iEpoch)

        elif params['chunkSize'] == 0:
            feedDict = {tfFeat: dfFeat * params['featScalerA'] + params['featScalerB'],
                        tfEngy: dfEngy * params['engyScalerA'] + params['engyScalerB'],
                        tfLR: params['learningRate']}

            sess.run(tfOptimizer, feed_dict=feedDict)
        else:
            print("Invalid chunkSize, not within [0,inf]. chunkSize=", params['chunkSize'])

        if iEpoch % 10 == 0:
            Ep, loss = sess.run((tfEs, tfLoss), feed_dict=feedDict)
            Ep = (Ep - params['engyScalerB']) / params['engyScalerA']
            Ermse = np.sqrt(np.mean((Ep - dfEngy) ** 2))
            print(iEpoch, loss, Ermse)
            sys.stdout.flush()

        if params["validate"] > 0:
            if iEpoch % params["validate"] == 0:
                print(str(iEpoch) + "th epoch")
                getError('v' + str(params['featFile']), 'v' + str(params['engyFile']))
        if params["test"] > 0:
            if iEpoch % params["test"] == 0:
                print(str(iEpoch) + "th epoch")
                getError('t' + str(params['featFile']), 't' + str(params['engyFile']))

    if params["validate"] == 0:
        getError('v' + str(params['featFile']), 'v' + str(params['engyFile']))
    if params["test"] == 0:
        getError('t' + str(params['featFile']), 't' + str(params['engyFile']))

    save_path = saver.save(sess, str(params['logDir']) + "/tf.chpt")
    return save_path

'''

def old_specialTask01(params):
    #   Generate features for the Gaussian symmetry functions
    #
    Rc = 6.2
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size

    featFile = str(params["featFile"])
    engyFile = str(params["engyFile"])

    nCase = 0
    with open(str(params["inputData"]), 'r') as datafile:
        for line in datafile:
            if "Iteration" in line:
                nCase += 1

    with open(str(params["inputData"]), 'r') as datafile:
        for i in range(nCase):
            nAtoms, iIter, lattice, R, f, v, e = pyf.getData(datafile)
            idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, float(params["dcut"]))
            Rhat, Ri, Dc = npf.getStruct(coord)
            Di = Ri[:,:,None] * np.ones(256)
            Di[Dc == 0] = 0
            Dj = np.transpose(Di, [0,2,1])
            feat = getGaussianFeats(ll, eta, zeta, Rs, Ri, Di, Dj, Dc)
            engy = e.reshape([-1, 1])
            pd.DataFrame(feat).to_csv(featFile, mode='a', header=False, index=False)
            pd.DataFrame(engy).to_csv(engyFile, mode='a', header=False, index=False)


def old_specialTask02(params):
    #   Train for the Gaussian basis
    #

    Rc = 6.2
    ll = np.array([-1., 1])
    eta = np.arange(1, 6) ** 2 / Rc ** 2
    zeta = np.arange(0, 6)
    Rs = np.arange(1, 11) * Rc / 10
    num2bFeat = Rs.size * eta.size  # number of 2body features
    num3bFeat = ll.size * eta.size * zeta.size
    numFeat = num2bFeat + num3bFeat

    #   NN

    tfFeat = tf.placeholder(tf.float32, shape=(None, numFeat))
    tfEngy = tf.placeholder(tf.float32, shape=(None, 1))
    tfLR = tf.placeholder(tf.float32)

    tfEs = tff.tf_engyFromFeats(tfFeat, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    tfLoss = tf.reduce_mean((tfEs - tfEngy) ** 2)

    return 0

'''
