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
stop=30
alpha = 1.0 # Alpha should NOT be an integer (add .0 at the end to avoid rounding)
mu = 1.0 # Mu should NOT be an integer (add .0 at the end to avoid rounding)
          # Use mu and set alpha=1.0 for backtracking
myLambda = 3
res = 1000

draw_error = True
draw_functions = not draw_error
draw_iterates = not draw_error

points, X, Y, points_ref, X_ref, Y_ref = generate_signal.generate(n, p, res=res)

# Fista
g = primal_elanet.ElaNet_Function(myLambda, myLambda*alpha/mu)
f = primal_elanet.Square_Loss_Function(X, Y)
A, B, C = np.linalg.svd(X) # This is cheating (svd costs n^3, more than our entire algo)
print 'Svd - done'
grad_lip_cte = 2*B[0]**2
reg_path2 = prox_descent.prox_descent(2*p, f, g, stop, 1/grad_lip_cte)
print 'Fista - done'

# Primal dual splitting
grad_lip_cte = B[0]**2/alpha
f = elanet_new_version.L1_Norm(mu)
g = elanet_new_version.Scalar_Product(Y)
operator = elanet_new_version.mult(X)
reg_path3, diff = reg_prox_descent.reg_prox_descent((2*p,1),(n,1), f, g, operator,
                        alpha, stop, -1, np.zeros((2*p,1)))
print "Primal dual splitting - done"

reg_path1, diff = reg_prox_descent.reg_prox_descent((2*p,1),(n,1), 
                        f, g, operator, alpha, stop, 1/grad_lip_cte, np.zeros((2*p,1)))
print "Accelerated primal dual splitting - done"

score1 = np.zeros(stop)
score2 = np.zeros(stop)
score3 = np.zeros(stop)

val1 = np.zeros(stop)
val2 = np.zeros(stop)
val3 = np.zeros(stop)


for i in range (stop):
    score1[i] = np.linalg.norm(Y_ref - np.dot(X_ref, reg_path1[:,:,i]))/ float(res)
    score2[i] = np.linalg.norm(Y_ref - np.dot(X_ref, reg_path2[:,i])) / float(res)
    score3[i] = np.linalg.norm(Y_ref - np.dot(X_ref, reg_path3[:,:,i]))/ float(res)
    val1[i] = np.linalg.norm(Y - np.dot(X, reg_path1[:,:,i]))/ float(n)
    val2[i] = np.linalg.norm(Y - np.dot(X, reg_path2[:,i]))/ float(n)
    val3[i] = np.linalg.norm(Y - np.dot(X, reg_path3[:,:,i]))/ float(n)
    if (draw_iterates):
        plt.plot(points_ref, np.dot(X_ref, reg_path1[:,:,stop-1-5]), 'y-')
        plt.plot(points_ref, np.dot(X_ref, reg_path3[:,:,stop-1-5]), 'b-')
        plt.plot(points_ref, np.dot(X_ref, reg_path2[:,stop-1-5]), 'r-')
  

if (draw_error):
    plt.plot(range(stop), score1, 'y', linewidth = 2) 
    #plt.plot(range(stop), score2, 'r', linewidth = 2) 
    plt.plot(range(stop), score3, 'g', linewidth = 2) 
    
    plt.plot(range(stop), val1, 'y') 
    #plt.plot(range(stop), val2, 'r') 
    plt.plot(range(stop), val3, 'g') 

if (draw_functions):
    plt.plot(points_ref, np.dot(X_ref, reg_path3[:,:,stop-1]), 'b', linewidth = 2)
    plt.plot(points_ref, np.dot(X_ref, reg_path2[:,stop-1]), 'r', linewidth = 2)
    plt.plot(points_ref, np.dot(X_ref, reg_path1[:,:,stop-1]), 'y-', linewidth = 2)
    plt.plot(points, Y, 'p')


