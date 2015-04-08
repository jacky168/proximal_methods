# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:42:58 2015

@author: Simon Matet

"""

import numpy as np
import copy

""" Interface for g function """
class Base_g_Function (object):
    # Replace this by real proximal
    def prox(self):
        raise NotImplementedError("Should have implemented proximal operator")
        
""" Interface for g function """
class Base_f_Function (object):
    # Replace this by real proximal
    def grad(self):
        raise NotImplementedError("Should have implemented proximal operator")
    
    def value(self):
        raise NotImplementedError("Should have implemented proximal operator")
        
def default_eta(k):
    return 1  
       
       
"""  Proximal Descent Method per se
     Minimizes f(x) + g(x) ; f convex differentiable ; g convex
     Arguments :
         - p          : integer : number of covariates
         - f function : object with method grad and value 
                 • Value is optionnal if grad_lip_cte is non negative.
                 • Should inherit Base_f_Function
         - g function : object with method prox
                 • Should inherit Base_g_Function
         - stop       : integer : number of iterations
         - (optional) grad_lip_cte : default = -1 for unknown (does a line search, need f.value)
         - (optional) eta : averaging factor ; default = 1
         - (optional) ini : initialiation value ; default = 0
"""
def prox_descent (p, f, g, stop, grad_lip_cte=-1, eta=default_eta,ini=0):
    if (ini==0):
       v1 = np.transpose(np.matrix(np.zeros(p)))
    else:
        v1 = ini
            
    reg_path = np.matrix(np.zeros((p, stop)))
    step = 1000        
    for i in range (stop):
        current_grad = f.grad(v1)
        if (grad_lip_cte >= 0): # If we know the lipschitz constante
            step = 1/grad_lip_cte
            v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step, v1 - step * current_grad)
        else: # If we don't : Line search ; to be implemented ; not working for now
            current_f_value = f.value(v1)
            step = 1000        
            while (True):
                v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step, v1 - step * current_grad)
                if (f.value(v2) <= current_f_value + np.dot(np.transpose(current_grad), v2 - v1) + 1/2/step * np.linalg.norm(v1 - v2)**2):
                    break
                else:
                    step = step * 0.8
        v1 = v2     
        reg_path[:,i] = v1
        
    return reg_path
    
def accelerated_prox_descent (p, f, g, stop, grad_lip_cte=-1, eta=default_eta,ini=0):
    if (ini==0):
        v2 = np.transpose(np.matrix(np.zeros(p)))
        v1 = np.transpose(np.matrix(np.zeros(p)))
    else:
        v2 = ini
        v1 = np.transpose(np.matrix(np.zeros(p)))
    
    reg_path = np.matrix(np.zeros((p, stop)))
    step = 1000        
    for i in range (stop):
        y = v2 + float(i)/float(i+3) * (v2 - v1)
        print np.sum(np.abs((v2 - v1)))
        v1 = v2
        
        current_grad = f.grad(y)
        if (grad_lip_cte >= 0): # If we know the lipschitz constante
            step = 1/grad_lip_cte
            v2 = (1 - eta(i)) * y + eta(i) * g.prox(step, y - step * current_grad)
        else: # If we don't : Line search ; to be implemented ; not working for now
            current_f_value = f.value(y)
            step = 1000        
            while (True):
                v2 = (1 - eta(i)) * y + eta(i) * g.prox(step, y - step * current_grad)
                if (f.value(v2) <= current_f_value + np.dot(np.transpose(current_grad), v2 - y) + 1/2/step * np.linalg.norm(y - v2)**2):
                    break
                else:
                    step = step * 0.8
        
        reg_path[:,i] = v2

    return reg_path