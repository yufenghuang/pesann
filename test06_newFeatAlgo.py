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

dcut = 6.2

tfCoord = tf.placeholder(tf.float32, shape=(None, 3))
tfLattice = tf.placeholder(tf.float32, shape=(3, 3))

tfFeatA = tf.constant(params['featScalerA'], dtype=tf.float32)
tfFeatB = tf.constant(params['featScalerB'], dtype=tf.float32)
numFeat = params['n2bBasis'] + params['n3bBasis'] ** 3

tfIdxNb, tfRNb, tfMaxNb, tfNAtoms = tff.tf_getNb(tfCoord, tfLattice, float(params['dcut']))
tfRhat, tfRi, tfDc = tff.tf_getStruct(tfRNb)

tfDc = tf.where(tfDc < float(params['dcut']), tfDc, tf.zeros_like(tfDc))

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

tfFeats, tfdXi, tfdXin = tff.tf_get_dXidRl2(tfGR2, tfGR2d, tfGR3, tfGR3d, tfGD3, tfRhat * RcA)
tfdXi = tf.expand_dims(tfFeatA, 2) * tfdXi
tfdXin = tf.expand_dims(tfFeatA, 2) * tfdXin

# tfFeats = tfFeatA * tff.tf_getFeats(tfGR2, tfGR3, tfGD3) + tfFeatB
tfFeats = tfFeatA * tfFeats + tfFeatB
tfEs = tff.tf_engyFromFeats(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])

dEldXi = tff.tf_get_dEldXi(tfFeats, numFeat, params['nL1Nodes'], params['nL2Nodes'])
Fll = tf.squeeze(tf.matmul(tf.expand_dims(dEldXi, 1), tfdXi))

dENldXi = tf.gather_nd(dEldXi,
                       tf.expand_dims(tf.transpose(tf.boolean_mask(tfIdxNb, tf.greater(tfIdxNb, 0)) - 1), 1))
dEnldXin = tf.scatter_nd(tf.where(tf.greater(tfIdxNb, 0)), dENldXi, [tfNAtoms, tfMaxNb, numFeat])
Fln = tf.squeeze(tf.matmul(tf.expand_dims(dEnldXin, 2), tfdXin))

tfEngyA = tf.constant(params['engyScalerA'], dtype=tf.float32)
tfEngyB = tf.constant(params['engyScalerB'], dtype=tf.float32)
tfFs1, tfFs2 = Fll, Fln

tfEp = (tfEs - tfEngyB) / tfEngyA
tfFp = (tfFs1 + tf.reduce_sum(tfFs2, axis=1)) / tfEngyA
tfFpq = tfFs2 / tfEngyA

saver = tf.train.Saver(list(set(tf.get_collection("saved_params"))))

nAtoms, lattice, R, V, F = mdf.readXYZ("md.xyz")
R = np.linalg.solve(lattice, R.T).T
R = R - np.floor(R)

feedDict = {tfCoord: R, tfLattice: lattice}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, str(params['logDir']) + "/tf.chpt")
    Ei, Fi = sess.run((tfEp, tfFp), feed_dict=feedDict)
    # dXi, dXin = sess.run((tfdXi, tfdXin), feed_dict=feedDict)