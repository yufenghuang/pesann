import numpy as np
import py_func as pyf
import np_func as npf
import matplotlib.pyplot as plt

def read_xyz(xyzFile):
    nAtoms = int(xyzFile.readline())
    xyzFile.readline()
    R = np.zeros((nAtoms, 3))
    for i in range(nAtoms):
        R[i,:] = np.array(xyzFile.readline().split()[1:4], dtype=float)
    return nAtoms, R

def getXYZRiDc(xyzFilename, lattice):
    with open(xyzFilename, 'r') as xyzFile:
        nAtoms, R0 = read_xyz(xyzFile)
    R = np.linalg.solve(lattice, R0.T).T
    idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, 6.2)
    Rhat, Ri, Dc = npf.getStruct(coord)
    return Ri, Dc

def getMMTRiDc(mmtFilename):
    with open(mmtFilename, 'r') as mmtFile:
        nAtoms, iIter, lattice, R, forces, velocities, energies = pyf.getData(mmtFile)
    idxNb, coord, maxNb, nAtoms = npf.np_getNb(R, lattice, 6.2)
    Rhat, Ri, Dc = npf.getStruct(coord)
    return Ri, Dc, lattice

def getAngles(Ri, Dc):
    Rj = Ri[:, :, None] * np.ones(Ri.shape[-1])
    Rk = Rj.transpose([0, 2, 1])
    cos = (Dc[Dc > 0] ** 2 - Rj[Dc > 0] ** 2 - Rk[Dc > 0] ** 2) / (2 * Rj[Dc > 0] * Rk[Dc > 0])
    angles = np.zeros_like(Dc)
    angles[Dc>0] = np.arccos(cos) * 180 / np.pi
    return angles


mmtFilename = "repul_xyz/MOVEMENT.Oct.17_50fs"
xyz0fs = "repul_xyz/nn_0fs.xyz"
xyz50fs = "repul_xyz/nn_50fs.xyz"
xyz100fs = "repul_xyz/nn_100fs.xyz"
xyz200fs = "repul_xyz/nn_200fs.xyz"
xyz500fs = "repul_xyz/nn_500fs.xyz"
xyz1000fs = "repul_xyz/nn_1000fs.xyz"

Ri, Dc, lattice = getMMTRiDc(mmtFilename)
angles = getAngles(Ri, Dc)
plt.figure()
plt.title("MOVEMENT.Oct.17 at 50fs")
plt.xlabel("R(Angstroms)")
plt.hist(Ri[Ri>0],50)

plt.figure()
plt.title("MOVEMENT.Oct.17 at 50fs")
plt.xlabel("Angles(degress)")
plt.hist(angles[Dc>0], 50)

Ri, Dc = getXYZRiDc(xyz0fs, lattice)
angles = getAngles(Ri, Dc)
plt.figure()
plt.title("NN MD at 0fs")
plt.xlabel("R(Angstroms)")
plt.hist(Ri[Ri>0],50)

plt.figure()
plt.title("NN MD at 0fs")
plt.xlabel("Angles(degress)")
plt.hist(angles[Dc>0], 50)

Ri, Dc = getXYZRiDc(xyz500fs, lattice)
angles = getAngles(Ri, Dc)
plt.figure()
plt.title("NN MD at 500fs")
plt.xlabel("R(Angstroms)")
plt.hist(Ri[Ri>0],50)

plt.figure()
plt.title("NN MD at 500fs")
plt.xlabel("Angles(degress)")
plt.hist(angles[Dc>0], 50)


Ri, Dc = getXYZRiDc(xyz1000fs, lattice)
angles = getAngles(Ri, Dc)
plt.figure()
plt.title("NN MD at 1000fs")
plt.xlabel("R(Angstroms)")
plt.hist(Ri[Ri>0],50)

plt.figure()
plt.title("NN MD at 1000fs")
plt.xlabel("Angles(degress)")
plt.hist(angles[Dc>0], 50)


# Rj = Ri[:,:,None] * np.ones(Ri.shape[-1])
# Rk = Rj.transpose([0,2,1])
# cos = (Dc[Dc>0]**2 - Rj[Dc>0]**2 - Rk[Dc>0]**2)/(2*Rj[Dc>0]*Rk[Dc>0])
# angles = np.arccos(cos)*180/np.pi

# plt.figure()
# plt.hist(angles, 50)