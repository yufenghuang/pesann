import md_func as mdf
import np_func as npf
import numpy as np
import py_func as pyf
import time
import tensorflow as tf
import tf_func as tff

params={
    "task":1,
    "chunkSize": 0,
    "epoch": 5000,
    "restart": False,
    "inputData": "md.xyz",
    "format": "mmt",
    "featFile": "feat",
    "engyFile": "engy",
    "logDir": "log",
    "iGPU": 0,
    "dcut": 6.2,
    "Rcut": 0,
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
    "nstep":1, # print every 1 steps
    "repulsion":"None",
    "mmtForces":"None",
    "T": 0,
    "Tbegin": -1,
    "Tend": -1,
    "dTstep": 0,
    "coll_prob": 0,
    }


newParams={}
floatParams={"dcut", "Rcut"}
intParams={"n2bBasis", "n3bBasis", "nL1Nodes", "nL2Nodes"}
savedScaler={"featScalerA", "featScalerB", "engyScalerA", "engyScalerB"}

paramFile = str(params['logDir']) + "/params"
loadParams = np.load(paramFile + ".npz")

for param in floatParams:
    params[param] = float(loadParams[param])
    if param in set(newParams.keys()):
        params[param] = float(newParams[param])

for param in intParams:
    params[param] = int(loadParams[param])
    if param in set(newParams.keys()):
        params[param] = int(newParams[param])

for param in savedScaler:
    params[param] = loadParams[param]


def tf_getEFln(tfCoord, tfLattice, params):
    tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
    tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
    numFeat = params['n2bBasis'] + params['n3bBasis'] ** 3

    tfIdxMat, tfIdxNb, tfRNb, tfMaxNb, tfNAtoms = tff.tf_getNb2(tfCoord, tfLattice, float(params['dcut']))

    tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)

    tfDc = tf.where(tfDc < float(params['dcut']), tfDc, tf.zeros_like(tfDc))

    tfRiMat = tf.scatter_nd(tf.where(tfIdxMat > 0), tf.boolean_mask(tfRi, tfIdxNb>0), tf.shape(tfIdxMat, out_type=tf.int64))

    tfIdxNb2 = tf.tile(tfIdxNb[:,None,:], [1, tf.cast(tfMaxNb, tf.int32), 1])
    tfIdxNb3 = tf.transpose(tfIdxNb2, [0,2,1])

    tf_idx1 = tf.boolean_mask(tfIdxNb2, (tfIdxNb2 > 0) & (tfIdxNb3 > 0)) - 1
    tf_idx2 = tf.boolean_mask(tfIdxNb3, (tfIdxNb2 > 0) & (tfIdxNb3 > 0)) - 1

    tfRiList = tf.gather_nd(tfRiMat, tf.transpose(tf.stack([tf_idx1, tf_idx2])))
    tfDi = tf.scatter_nd(tf.where((tfIdxNb2>0) & (tfIdxNb3>0)), tfRiList, tf.shape(tfIdxNb2, out_type=tf.int64))


    RcA = 2 / (float(params['dcut']) - float(params['Rcut']))
    RcB = - (float(params['dcut']) + float(params['Rcut'])) / (float(params['dcut']) - float(params['Rcut']))

    tfGR2 = tf.scatter_nd(tf.where(tfRi > 0),
                          tff.tf_getCos2(tf.boolean_mask(tfRi, tfRi > 0) * RcA + RcB, params['n2bBasis']),
                          [tfNAtoms, tfMaxNb, params['n2bBasis']])
    tfGR2d = tf.scatter_nd(tf.where(tfRi > 0),
                           tff.tf_getdCos2(tf.boolean_mask(tfRi, tfRi > 0) * RcA + RcB, params['n2bBasis']),
                           [tfNAtoms, tfMaxNb, params['n2bBasis']])
    tfGR3 = tf.scatter_nd(tf.where(tfRi > 0),
                          tff.tf_getCos2(tf.boolean_mask(tfRi, tfRi > 0) * RcA + RcB, params['n3bBasis']),
                          [tfNAtoms, tfMaxNb, params['n3bBasis']])
    tfGR3d = tf.scatter_nd(tf.where(tfRi > 0),
                           tff.tf_getdCos2(tf.boolean_mask(tfRi, tfRi > 0) * RcA + RcB, params['n3bBasis']),
                           [tfNAtoms, tfMaxNb, params['n3bBasis']])
    tfGD3 = tf.scatter_nd(tf.where(tfDc > 0),
                          tff.tf_getCos2(tf.boolean_mask(tfDc, tfDc > 0) * RcA + RcB, params['n3bBasis']),
                          [tfNAtoms, tfMaxNb, tfMaxNb, params['n3bBasis']])

    tfdXi, tfdXin = tff.tf_get_dXidRl(tfGR2, tfGR2d, tfGR3, tfGR3d, tfGD3, tfRhat * RcA)
    tfdXi = tf.expand_dims(tfFeatA, 2) * tfdXi
    tfdXin = tf.expand_dims(tfFeatA, 2) * tfdXin

    tfFeats = tfFeatA * tff.tf_getFeats(tfGR2, tfGR3, tfGD3) + tfFeatB
    tfEs = tff.tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])

    dEldXi = tff.tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
    Fll = tf.reduce_sum(tf.expand_dims(dEldXi, 2) * tfdXi, axis=1)

    dENldXi = tf.gather_nd(dEldXi,
                           tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb, 0)) - 1), 1))
    dEnldXin = tf.scatter_nd(tf.where(tf.greater(tfIdxNb, 0)), dENldXi, [tfNAtoms, tfMaxNb, numFeat])
    Fln = tf.reduce_sum(tf.expand_dims(dEnldXin, 3) * tfdXin, axis=2)

    return tfEs, Fll, Fln, tfDi - tfDc


nAtoms, lattice, R, V, F = mdf.readXYZ("md.xyz")
R = np.linalg.solve(lattice, R.T).T
R = R - np.floor(R)

dcut = 6.2

tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
tfLattice = tf.placeholder(tf.float32, shape=(3, 3))
tfTemp, tf_idxNb, tf_RNb, tf_maxNb, tf_nAtoms = tff. tf_getNb2(tfCoord, tfLattice, dcut)
_, __, ___, tfTemp2 = tf_getEFln(tfCoord, tfLattice, params)

tf_temp1 = tf.tile(tf.ones((3,4), dtype=tf.float32)[:,None,:], [1,3,1])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    feedDict = {tfCoord: R, tfLattice: lattice}
    idxNb, coord, maxNb, nAtoms = sess.run((tf_idxNb, tf_RNb, tf_maxNb, tf_nAtoms), feed_dict=feedDict)
    temp2 = sess.run(tfTemp2, feed_dict=feedDict)
    temp1 = sess.run(tf_temp1, feed_dict=feedDict)

idxMat = npf.adjList2adjMat(idxNb)

idxNb2 = idxNb[:, None, :] * np.ones((maxNb, 1), dtype=int)
idxNb3 = idxNb2.transpose([0, 2, 1])
