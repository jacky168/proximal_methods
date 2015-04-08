# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:54:02 2015

@author: user
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import reg_prox_descent
import low_rank_estimation
import test_images

rank = 10
dim1, dim2 = 300,300
stop = 600
epsilon = 0.00001
noise = 0.000001

#Mref = test_images.get_low_rank_image(rank)
A = np.random.normal(0.0, 1.0, (dim1,rank))
B = np.random.normal(0.0, 1.0, (rank,dim2))
Mref = np.dot(A,B)
dim1, dim2 = Mref.shape

M = Mref #+ np.random.normal(loc=0.0, scale = noise, size = (dim1, dim2))
mask = np.random.binomial(1,0.95,size=(dim1,dim2))
M = np.multiply(M, mask)

iniRandom = M + np.multiply(1-mask,np.random.normal(size=(dim1,dim2)))

"""
plt.imshow(Mref,cmap = cm.Greys_r)
plt.imshow(M,cmap = cm.Greys_r)
"""

f = low_rank_estimation.Nuclear_Norm()
g = low_rank_estimation.Scalar_Product(M)
proj = low_rank_estimation.Projection_On_Known_Positions(mask)

W = reg_prox_descent.reg_prox_descent((dim1, dim2),(dim1, dim2) , f, g, proj, epsilon, stop, 1/epsilon, M, ini=iniRandom)

norms = np.zeros(stop)
norms2 = np.zeros(stop)
relative_error = np.zeros(stop)
for i in range (stop):
    norms[i] = np.linalg.norm(np.multiply(W[:,:,i] - M,mask))
    norms2[i] = np.linalg.norm(Mref - W[:,:,i])
    relative_error[i] = np.linalg.norm(Mref - W[:,:,i]) / np.linalg.norm(Mref)
   
   
plt.plot(np.log(range(stop)), np.log(relative_error))

"""
plt.plot(range(stop), norms, 'b')
plt.plot(range(stop), norms2, 'r')
plt.plot(range(stop), relative_error, 'g')
print relative_error
"""