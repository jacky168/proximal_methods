# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:32:37 2015

@author: Simon Matet

Test file

"""

import numpy as np
import matplotlib.pyplot as plt

import generate_signal
import prox_descent
import primal_elanet
import elanet_new_version
import reg_prox_descent
import dual_elanet

n=100
p=200
stop=15
alpha = 0.01

points, X, Y, points_ref, X_ref, Y_ref = generate_signal.generate(n, p)

f = dual_elanet.ElaNet_Dual_With_Operator(X, alpha)
g = dual_elanet.Scalar_Product(Y)
A, B, C = np.linalg.svd(X)
grad_lip_cte = B[0]**2 /alpha# Largest singular value of the kernel

"""
reg_path = prox_descent.prox_descent(n, f, g, stop, grad_lip_cte)
buff = np.dot(np.transpose(X), reg_path)
reg_path = np.clip(-1*buff - 1/alpha, 0, np.inf) - np.clip (buff -1/alpha, 0, np.inf)
"""
"""
reg_path3 = prox_descent.accelerated_prox_descent(n, f, g, stop, grad_lip_cte)
buff = np.dot(np.transpose(X), reg_path3)
reg_path3 = np.clip(-1*buff - 1/alpha, 0, np.inf) - np.clip (buff -1/alpha, 0, np.inf)

g = primal_elanet.ElaNet_Function(10, 10*alpha)
f = primal_elanet.Square_Loss_Function(X, Y)
A, B, C = np.linalg.svd(X)
grad_lip_cte = 2*B[0]**2 # Largest singular value of the kernel
reg_path2 = prox_descent.prox_descent(2*p, f, g, stop, grad_lip_cte)
"""

f = elanet_new_version.L1_Norm()
g = elanet_new_version.Scalar_Product(Y)
operator = elanet_new_version.mult(X)
reg_path3, diff = reg_prox_descent.accelerated_reg_prox_descent((2*p,1),(n,1), f, g, operator, alpha, stop, 1/grad_lip_cte, np.zeros((2*p,1)))
print np.linalg.norm(np.dot(X, reg_path3[:,:,stop-1]) - Y)

"""
a = np.zeros(stop)
for i in range(stop):
    a[i] = np.linalg.norm(np.dot(X, reg_path3[:,:,i]) - Y)
    
plt.plot(np.log(range(stop)),np.log(a))
"""

for i in range (stop):
   plt.plot(points_ref, np.dot(X_ref, reg_path3[:,:,i]), 'b')
   # plt.plot(points_ref, np.dot(X_ref, reg_path2[:,i]), 'g')
    

