import md_func as mdf
import np_func as npf
import numpy as np
import time
import tensorflow as tf

nAtoms, lattice, R, V, F = mdf.readXYZ("md.xyz")
R = np.linalg.solve(lattice, R.T).T
R = R - np.floor(R)

dcut = 6.2

idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, dcut)
Rhat, Ri, Dc = npf.getStruct(coord)

Di = np.zeros_like(Dc)

idxMat, RiMat = npf.adjList2adjMat(idxNb, Ri)

idxNb2 = idxNb[:,None,:] * np.ones((maxNb,1), dtype=int)
idxNb3 = idxNb2.transpose([0,2,1])

Dc[Dc>dcut] = 0
Di[(idxNb2>0) & (idxNb3>0)] = RiMat[idxNb2[(idxNb2>0) & (idxNb3>0)]-1, idxNb3[(idxNb2>0) & (idxNb3>0)]-1]
i,j,k = np.where(np.abs(Di - Dc) > 0.00001)

t = time.time()
np.matmul(Ri[:,None,:], Dc)
print(time.time()-t)

t = time.time()
np.sum(Ri[:,None,:] * Dc, axis=2)
print(time.time()-t)

t = time.time()
Dtemp = np.zeros_like(Dc)
Dtemp[Ri>0] = (Ri[Ri>0])[:,None] * Dc[Ri>0]
Dtemp.sum(axis=1)
print(time.time()-t)

idxNb4 = np.zeros((maxNb, nAtoms, maxNb), dtype=int)
ones = np.ones(maxNb,dtype=int)
idxNb4[ones>0] = idxNb

tfx = tf.constant(np.arange(12).reshape(3,4), shape=(3,4), dtype=tf.float32)
tfy = tf.tile(tfx[None,:,:], [4,1,1])
with tf.Session() as sess:
    y = sess.run(tfy)
