# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:59:01 2018

@author: hif10
"""

import re
import numpy as np

def getData(dataFile):
    line = dataFile.readline()
    if "Iteration" in line:
        sptline = line.split()
        print(sptline)
        nAtoms = int(sptline[0])
        iIter = int(re.match("(\d*),",sptline[2])[1])
        iEp = float(sptline[5])
        return nAtoms, iIter, iEp
    else:
        return getData(dataFile)
