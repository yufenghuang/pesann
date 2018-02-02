#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:49:17 2018

@author: yufeng
"""
import numpy as np
import py_func as pyf
import os
#import sys

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--runtype", choices=[-2,-1,0,1,2], type=int, default=1,\
                    help="Runtype. 2=get energy and forces, 1=get energy, 0=MD, -1=train energy, -2=train energy and forces")
parser.add_argument("--restart", action="store_true", help="Restarting the calculations with old parameters")
parser.add_argument("--chunkSize", type=int, default=0)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--inputData", type=str)
parser.add_argument("--logDir", type=str, default="log")
parser.add_argument("--iGPU", type=int, default=0)
parser.add_argument("--feRatio", type=float, default=1.0)
parser.add_argument("--dcut", type=float, default=6.2)
parser.add_argument("--learningRate", type=float, default=0.0001)
parser.add_argument("--n2bBasis", type=int, default=100)
parser.add_argument("--n3bBasis", type=int, default=10)
parser.add_argument("--nL1Nodes", type=int, default=300)
parser.add_argument("--nL2Nodes", type=int, default=500)

args = parser.parse_args()
for arg in vars(args):
    print(arg, "=",getattr(args, arg))

#sys.exit()

#import re

'''
sampleInput = "# comments \n" \
              "a = 1 \n" \
              "b = \"movement file\" # specifying b \n" \
              "c = # wrong input \n"

for line in sampleInput.split("\n"):
    line0 = line.split('#')[0].strip()
    if line0 != '':
        if re.match("^([\w\s\"]+)=([\w\s\"]+)$", line0):
            key = line0.split("=")[0].strip().strip("\"")
            value = line0.split("=")[1].strip().strip("\"")
            print(key,"is",value)
        else:
            print("unknown input", line0)
'''



params = {
        "chunkSize": 0,
        "epoch": 10,
        "restart": True,
        "inputData": "MOVEMENT.train.first100",
        "featFile": "feat",
        "engyFile": "engy",
        "logDir": "log",
        "iGPU": 0,
        "runtype": -2,  # 2: evaluate energy and forces
                        # 1: evaluate energy
                        # 0: MD
                        #-1: training with energy
                        #-2: training with energy and forces
        "mmtFile": "coord.mmt",
        "feRatio": 1,
        }

params["runtype"] = args.runtype
params["inputData"] = args.inputData

os.environ["CUDA_VISIBLE_DEVICES"]=str(params['iGPU'])

if params['restart']:
    paramFile = str(params['logDir'])+"/params"
    loadParams = np.load(paramFile+".npz")
    oldParams = {}
    for param in loadParams.files:
        oldParams[param] = loadParams[param]
    for param in params:
        oldParams[param] = params[param]
    params = oldParams
else:
    params = pyf.initialize(params)

print("Initialization done")

if params["runtype"] == 2:
    Ep,Fp = pyf.getEngyFors(params)
elif params["runtype"] == 1:
    Ep = pyf.getEngy(params)
elif params["runtype"] == 0:
    pass
elif params["runtype"] == -1:
    params["featFile"] = "feat"
    params["engyFile"] = "engy"
    pyf.outputFeatures(params)
    print(pyf.trainEngy(params))
elif params["runtype"] == -2:
    print(pyf.trainEF(params))
else:
    print("Unrecognized runtype: ", params["runtype"])
    