#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:19:30 2018

@author: yufeng
"""

import numpy as np
import pandas as pd
import tensorflow as tf

nTestSet = 100       # test set
nValidSet = 100      # validation set
nTrainSet = -1       # training set

nDataSet = np.array([nTestSet,nValidSet,nTrainSet])