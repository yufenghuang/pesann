#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:49:17 2018

@author: yufeng
"""
import numpy as np
import py_func as pyf
import os

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
    params["mmtFile"] = "coord.mmt"
    Ep,Fp = pyf.getEngyFors(params)
elif params["runtype"] == 1:
    params["mmtFile"] = "coord.mmt"
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
    