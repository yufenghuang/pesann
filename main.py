#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:49:17 2018

@author: yufeng
"""
import numpy as np
import py_func as pyf
import md
import specialTask
import os
import sys
import pandas as pd
import md_func as mdf

import argparse

params={
    "task":1,
    "chunkSize": 0,
    "epoch": 5000,
    "restart": False,
    "inputData": "MOVEMENT.train.first100",
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
    "fracMem": 0.2,
    }

newParams={}

floatParams={"dcut", "Rcut"}
intParams={"n2bBasis", "n3bBasis", "nL1Nodes", "nL2Nodes"}
savedScaler={"featScalerA", "featScalerB", "engyScalerA", "engyScalerB"}

parser = argparse.ArgumentParser()

parser.add_argument("--task", choices=[-3,-2,-1,0,1,2,100,105,106, 107, 108, 109,110,111, 112, 113,114,115,
                                       201, 202,203, 204, 205, 206, 207, 208, 209, 210, 212, 213,  301, 302, 303], type=int,
                    help="task.  2=get energy and forces, \
                                    1=get energy (default), \
                                    0=MD, \
                                   -1=generate features, \
                                   -2=train against E \
                                   -3=train against E & F")
parser.add_argument("inputData", type=str, help="Geometry file. Must include energies and/or forces when training")
parser.add_argument("--validationSet", type=str)
parser.add_argument("--testSet", type=str)

parser.add_argument("--inputFile", type=str, help="Input file specifying the calculation. \
                                                   Keys will be overwritten by command line arguments")
parser.add_argument("--format", type=str, choices=["xyz", "mmt"])
parser.add_argument("--restart", action="store_true",
                    help="Restart calculation by loading saved data  in the logDir directory. \
                    Seting this flag will ignore --dcut, --n2bBasis, --n3bBasis, --nL1Nodes and --nL2Nodes")
parser.add_argument("--chunkSize", type=int, help="default: chuckSize="+str(params["chunkSize"]))
parser.add_argument("--epoch", type=int, help="default: epoch="+str(params["epoch"]))
parser.add_argument("--logDir", type=str, help="default: logDir="+params["logDir"])
parser.add_argument("--iGPU", type=int, help="species which GPU to run the calculations")
parser.add_argument("--feRatio", type=float, help="the ratio between the energy term and the force term \
                                                   in the total loss function")
parser.add_argument("--learningRate", type=float, help="learning rate(step size) in the gradient descent optimization")
parser.add_argument("--dcut", type=float, help="outer cutoff radius for the calculation (Angstrom)")
parser.add_argument("--Rcut", type=float, help="inner cutoff radius for the calculation (Angstrom)")
parser.add_argument("--n2bBasis", metavar="N2B", type=int, help="number of basis for the 2D term")
parser.add_argument("--n3bBasis", metavar="N3B", type=int,
                    help="number of basis in each dimension for the 3D term, total number of basis equals N3B^3")
parser.add_argument("--nL1Nodes", type=int, help="number of nodes in the first hidden layer of the neural network")
parser.add_argument("--nL2Nodes", type=int, help="number of nodes in the second hidden layer of the neural network")
parser.add_argument("--validate", metavar="N_VALIDATE", type=int,
                    help="evaluate the errors on the validation set after every N_VALIDATE epochs")
parser.add_argument("--test", metavar="N_TEST", type=int,
                    help="evaluate the errors on the test set after every N_TEST epochs")
parser.add_argument("--dt", type=float, help="delta t, time step for the MD simulation. Unit is ps")
parser.add_argument("--nstep", type=int, help="dumping the geometry in xyz format in every NSTEP of MD simulations")

parser.add_argument("--repulsion", type=str, help="additional repulsion energy", choices=["None", "1/R12", "1/R","exp(-R)"])

parser.add_argument("--mmtForces", type=str, help="MOVEMENT file to train with forces")

parser.add_argument("--T", type=float, help="Inivitial temperature for the thermal conductivity calculation. Not yet working with MD")

parser.add_argument("--Tbegin", type=float, help="Initial temperature when applying the thermostat")
parser.add_argument("--Tend", type=float, help="Final temperature for the thermostat")
parser.add_argument("--dTstep", type=int, help="")
parser.add_argument("--coll_prob", type=float, help="")
parser.add_argument("--fracMem", type=float, help="fraction of the total GPU memory to be used")


args = parser.parse_args()

import re
if args.inputFile != None:
    with open(args.inputFile, 'r') as inputFile:
        for readline in inputFile:
#            line = readline.split("\n")
            line = readline.split('#')[0].strip()
            if line != '':
                if re.match("^([\w\s\"]+)=([\w\s\"]+)$", line):
                    key = line.split("=")[0].strip().strip("\"")
                    value = line.split("=")[1].strip().strip("\"")
                    params[key] = type(params[key])(value)
#                    print(key,"is",value)
                else:
                    print("unknown input", line)
                    print("exiting...")
                    sys.exit()

for arg in vars(args):
    if getattr(args, arg) != None:
        # print(arg, "=", getattr(args, arg))
        newParams[arg] = getattr(args, arg)
        params[arg] = getattr(args, arg)

os.environ["CUDA_VISIBLE_DEVICES"]=str(params['iGPU'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

if params['restart']:
    paramFile = str(params['logDir'])+"/params"
    loadParams = np.load(paramFile+".npz")

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

    # params["duct"] = float(loadParams["dcut"])
    # params["Rcut"] = float(loadParams["Rcut"])
    # params["n2bBasis"] = int(loadParams["n2bBasis"])
    # params["n3bBasis"] = int(loadParams["n3bBasis"])
    # params["nL1Nodes"] = int(loadParams["nL1Nodes"])
    # params["nL2Nodes"] = int(loadParams["nL2Nodes"])
    # params["featScalerA"] = loadParams["featScalerA"]
    # params["featScalerB"] = loadParams["featScalerB"]
    # params["engyScalerA"] = loadParams["engyScalerA"]
    # params["engyScalerB"] = loadParams["engyScalerB"]
else:
    params = pyf.initialize(params)

#print("==============")
for p in params:
    print(p, "=", params[p])


print("Initialization done")

if params["task"] == 2:
    Ep,Fp = pyf.getEngyFors(params)
    print("Total Energy is {}".format(np.sum(Ep)))
    print("Atomic energy + forces:")
    for i in range(len(Ep)):
        print("{} {} {} {}".format(Ep[i, 0], Fp[i, 0], Fp[i, 1], Fp[i, 2]))

elif params["task"] == 1:
    Ep = pyf.getEngy(params)
    print("Total Energy is {}".format(np.sum(Ep)))
    print("Atomic energy:")
    for i in range(len(Ep)):
        print("{}".format(Ep[i, 0]))

elif params["task"] == 0:
    pyf.NVE(params)
elif params["task"] == -1:
    if params["validationSet"] != "":
        pyf.outputFeatures("v"+params["engyFile"], "v"+params["featFile"], params["validationSet"], params)

    if params["testSet"] != "":
        pyf.outputFeatures("t"+params["engyFile"], "t"+params["featFile"], params["testSet"], params)

    pyf.outputFeatures(str(params["engyFile"]), str(params["featFile"]), str(params["inputData"]), params)
    # fFile = str(params["featFile"])
    # eFile = str(params["engyFile"])
    # dFile = str(params["inputData"])
    #
    # if params["validationSet"] != "":
    #     params["featFile"] = "v" + fFile
    #     params["engyFile"] = "v" + eFile
    #     params["inputData"] = params["validationSet"]
    #     pyf.outputFeatures(params)
    #
    # if params["testSet"] != "":
    #     params["featFile"] = "t" + fFile
    #     params["engyFile"] = "t" + eFile
    #     params["inputData"] = params["testSet"]
    #     pyf.outputFeatures(params)
    #
    # params["featFile"] = fFile
    # params["engyFile"] = eFile
    # params["inputData"] = dFile
    # pyf.outputFeatures(params)
elif params["task"] == -2:
    if params["validate"] >= 0:
        if (not os.path.exists("v"+str(params["featFile"]))) or \
           (not os.path.exists("v"+str(params["engyFile"]))):
            print("There are no features files for the validation set")
    
    if params["test"] >= 0:
        if (not os.path.exists("t"+str(params["featFile"]))) or \
           (not os.path.exists("t"+str(params["engyFile"]))):
            print("There are no features files for the test set")
            
    print(pyf.trainEngy(params))
    
elif params["task"] == -3:
    if (params["validate"] >= 0) & (not os.path.exists(str(params["validationSet"]))):
        print("You need to specify the validation set using --validationSet")
        
    if (params["validate"] < 0) & (os.path.exists(str(params["validationSet"]))):
        print("You might want to specify how frequent to calculate the errors \
              on the validation set using --validate")
        
    if (params["test"] >= 0) & (not os.path.exists(str(params["testSet"]))):
        print("You need to specify the test set using --testSet")
        
    if (params["test"] < 0) & (os.path.exists(str(params["testSet"]))):
        print("You might want to specify how frequent to calculate the errors \
              on the test set using --test")
    
    print(pyf.trainEF(params))
    
elif params["task"] == 100:
    md.specialrun4(params)
    
elif params["task"] == 105:
    md.specialrun5(params)

elif params["task"] == 106:
    md.specialrun6(params)

elif params["task"] == 107:

    md.specialrun7(params)

elif params["task"] == 108:

    md.specialrun8(params)

elif params["task"] == 109:

    md.specialrun9(params)

elif params["task"] == 110:

    md.specialrun10(params)

elif params["task"] == 111:

    md.specialrun11(params)

elif params["task"] == 112:

    md.specialrun12(params)

elif params["task"] == 113:

    md.specialrun13(params)

elif params["task"] == 114:

    md.specialrun14(params)

elif params["task"] == 115:

    md.specialrun15(params)

elif params["task"] == 201:

    if params["validationSet"] != "":
        specialTask.specialTask01("v"+params["engyFile"], "v"+params["featFile"], params["validationSet"], params)

    if params["testSet"] != "":
        specialTask.specialTask01("t"+params["engyFile"], "t"+params["featFile"], params["testSet"], params)

    specialTask.specialTask01(str(params["engyFile"]), str(params["featFile"]), str(params["inputData"]), params)

elif params["task"] == 202:

    if params["validate"] >= 0:
        if (not os.path.exists("v" + str(params["featFile"]))) or \
                (not os.path.exists("v" + str(params["engyFile"]))):
            print("There are no features files for the validation set")

    if params["test"] >= 0:
        if (not os.path.exists("t" + str(params["featFile"]))) or \
                (not os.path.exists("t" + str(params["engyFile"]))):
            print("There are no features files for the test set")

    if not params['restart']:
        pdFeat = pd.read_csv(str(params['featFile']), header=None, index_col=False,
                             chunksize=int(12800), iterator=True)
        pdEngy = pd.read_csv(str(params['engyFile']), header=None, index_col=False,
                             chunksize=int(12800), iterator=True)

        feat = pdFeat.get_chunk().values
        engy = pdEngy.get_chunk().values
        engy = engy.reshape((-1,1))

        params['featScalerA'], params['featScalerB'], params['engyScalerA'], params['engyScalerB'] = \
        pyf.getFeatEngyScaler(feat, engy)

        paramFile = str(params['logDir']) + "/params"
        np.savez(paramFile, **params)

    print(specialTask.specialTask02(params))

elif params["task"] == 203:

    specialTask.specialTask03(params)

elif params["task"] == 204:

    specialTask.specialTask04(params)

elif params["task"] == 205:

    specialTask.specialTask05(params)

elif params["task"] == 206:

    specialTask.specialTask06(params)

elif params["task"] == 207:

    specialTask.specialTask07(params)

elif params["task"] == 208:

    specialTask.specialTask08(params)

elif params["task"] == 209:

    specialTask.specialTask09(params)

elif params["task"] == 210:

    specialTask.specialTask10(params)

elif params["task"] == 212:

    specialTask.specialTask12(params)

elif params["task"] == 213:

    specialTask.specialTask13(params)

elif params["task"] == 301:

    mdf.andersen(params)

elif params["task"] == 302:

    mdf.NVE(params)

elif params["task"] == 303:

    mdf.hcacf(params)


else:
    print("Unrecognized task: ", params["task"])
