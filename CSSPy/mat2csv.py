#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:28:54 2019

@author: abelhadj
"""

import scipy.io
import numpy as np

data = scipy.io.loadmat("StrongRRQR/A.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("StrongRRQR/"+i+".csv"),data[i],delimiter=',')