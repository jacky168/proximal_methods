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
p=100
stop=10
alpha = 0.05
myLambda = n/4

points, X, Y, points_ref, X_ref, Y_ref = generate_signal.generate(n, p)

# Fista
g = primal_elanet.ElaNet_Function(myLambda, myLambda*alpha)
f = primal_elanet.Square_Loss_Function(X, Y)
A, B, C = np.linalg.svd(X) # This is cheating (svd costs n^3, more than our entire algo)
grad_lip_cte = 2*B[0]**2
reg_path2 = prox_descent.accelerated_prox_descent(2*p, f, g, stop, 1/grad_lip_cte)


# Primal dual splitting
grad_lip_cte = B[0]**2/alpha
f = elanet_new_version.L1_Norm()
g = elanet_new_version.Scalar_Product(Y)
operator = elanet_new_version.mult(X)
reg_path3, diff = reg_prox_descent.reg_prox_descent((2*p,1),(n,1), f, g, operator, alpha, stop, 1/grad_lip_cte, np.zeros((2*p,1)))


for i in range (stop):
   plt.plot(points_ref, np.dot(X_ref, reg_path3[:,:,i]), 'b-')
   plt.plot(points_ref, np.dot(X_ref, reg_path2[:,i]), 'r-')
   
plt.plot(points_ref, np.dot(X_ref, reg_path3[:,:,stop-1]), 'b', linewidth = 2)
plt.plot(points_ref, np.dot(X_ref, reg_path2[:,stop-1]), 'r', linewidth = 2)

plt.plot(points_ref, Y_ref, 'g', linewidth = 4)

