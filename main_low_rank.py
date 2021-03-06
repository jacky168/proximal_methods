# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:54:02 2015

@author: Simon Matet
"""

import numpy as np

import matplotlib.pyplot as plt

import Algos.reg_prox_descent as reg_prox_descent
import Functions.low_rank_estimation as low_rank_estimation

# When do you stop ??
# The epsilon question
# The initialization enigma
# The step_size_multiplier conundrum

rank = 50
dim1, dim2 = 1000,1000
epsilon = 1.0/5/dim1                                                    # Factor in front of the square norm
noise = 0.5                                                              # Ratio of noise
knowledge_ratio = 0.39                                                   # Probability that we know an element
step_size_multiplier = 1.2/knowledge_ratio                                # Because. Ask me.
#stop = int(np.power(1/noise/epsilon/step_size_multiplier, 2.0/3.0))     # Number of iterations (upper bound from theory)
stop = 150
print stop

# Creates the matrix
A = np.random.normal(0.0, 1.0, (dim1,rank))
B = np.random.normal(0.0, 1.0, (rank,dim2))
Mref = np.dot(A,B)

# Sampling + noise
M = Mref #+ np.random.normal(loc=0.0, scale = noise, size = (dim1, dim2))
mask = np.random.binomial(1,knowledge_ratio,size=(dim1,dim2))
M = np.multiply(M, mask)

# I tried initializing with missing values at random.
iniRandom = M + np.multiply(1-mask,np.random.normal(size=(dim1,dim2))) 

f = low_rank_estimation.Nuclear_Norm()
g = low_rank_estimation.Scalar_Product(M)
proj = low_rank_estimation.Projection_On_Known_Positions(mask)
comparator = low_rank_estimation.comparator(Mref)

# We can also try with accelerated version. W = whole "regularization path"
x, score = reg_prox_descent.reg_prox_descent((dim1, dim2),(dim1, dim2), f, g, 
                                            proj, epsilon, 
                                            stop, step_size_multiplier*epsilon, 
                                            M, comparator = comparator, ini=M, verbose = True,  mode=1)

score = score/np.linalg.norm(Mref)
                                            
plt.plot(range(stop), np.log(score), 'r')


