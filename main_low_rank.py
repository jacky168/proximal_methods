# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:54:02 2015

@author: Simon Matet
"""

import numpy as np

import matplotlib.pyplot as plt

import reg_prox_descent
import low_rank_estimation

# When do you stop ??
# The epsilon question
# The initialization enigma
# The step_size_multiplier conundrum

rank = 10
dim1, dim2 = 1000,1000
epsilon = 1.0/5/dim1                                                    # Factor in front of the square norm
noise = 1                                                               # Ratio of noise
knowledge_ratio = 0.5                                                   # Probability that we know an element
step_size_multiplier = 1/knowledge_ratio                                # Because. Ask me.
stop = int(np.power(1/noise/epsilon/step_size_multiplier, 2.0/3.0))     # Number of iterations (upper bound from theory)
stop = 100
print stop

# Creates the matrix
A = np.random.normal(0.0, 1.0, (dim1,rank))
B = np.random.normal(0.0, 1.0, (rank,dim2))
Mref = np.dot(A,B)

# Sampling + noise
M = Mref + np.random.normal(loc=0.0, scale = noise, size = (dim1, dim2))
mask = np.random.binomial(1,knowledge_ratio,size=(dim1,dim2))
M = np.multiply(M, mask)

# I tried initializing with missing values at random.
iniRandom = M + np.multiply(1-mask,np.random.normal(size=(dim1,dim2))) 

f = low_rank_estimation.Nuclear_Norm()
g = low_rank_estimation.Scalar_Product(M)
proj = low_rank_estimation.Projection_On_Known_Positions(mask)

# We can also try with accelerated version. W = whole "regularization path"
W, diff = reg_prox_descent.reg_prox_descent((dim1, dim2),(dim1, dim2), f, g, 
                                            proj, epsilon, 
                                            stop, step_size_multiplier*epsilon, 
                                            M, ini=M, verbose = True)

# Results
relative_error = np.zeros(stop)
distance_to_M = np.zeros(stop)
score = np.zeros(stop)
for i in range (stop):
    relative_error[i] = np.linalg.norm(Mref - W[:,:,i]) / np.linalg.norm(Mref)
    distance_to_M[i] = np.linalg.norm(M - proj.value(W[:,:,i])) / np.linalg.norm(M)
plt.plot(range(stop), np.log(relative_error))



# Various codes for results
"""
score = np.zeros(8)
for j in range(8):
    W, diff = reg_prox_descent.accelerated_reg_prox_descent((dim1, dim2),(dim1, dim2), f, g, proj, epsilon, stop, 2*(j+1)*epsilon, M, ini=M)
    relative_error2 = np.zeros(stop)
    distance_to_M2 = np.zeros(stop)
    score2 = np.zeros(stop)
    for i in range (stop):
        relative_error2[i] = np.linalg.norm(Mref - W[:,:,i]) / np.linalg.norm(Mref)
        distance_to_M2[i] = np.linalg.norm(M - proj.value(W[:,:,i])) / np.linalg.norm(M)
    
       
    #plt.plot(range(stop), np.log(relative_error), 'b')
    plt.plot(range(stop), np.log(relative_error2))
    
    score[j] = relative_error2[stop-1]
"""


"""
plt.plot(range(stop), norms, 'b')
plt.plot(range(stop), norms2, 'r')
plt.plot(range(stop), relative_error, 'g')
print relative_error
"""