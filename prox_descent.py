# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:42:58 2015

@author: Simon Matet

"""

import numpy as np

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
         - (optional) step_size : default = -1 for unknown (does a line search, need f.value)
         - (optional) eta : averaging factor ; default = 1
         - (optional) ini : initialiation value ; default = 0
"""
def prox_descent (p, f, g, stop, step_size, eta=default_eta,ini=0):
    if (ini==0):
       v1 = np.transpose(np.matrix(np.zeros(p)))
    else:
        v1 = ini
            
    reg_path = np.matrix(np.zeros((p, stop)))
    if (step_size < 0):
        step_size = 1000
    for i in range (stop):
        current_grad = f.grad(v1)
        if (step_size >= 0): # If we know the lipschitz constante
            v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step_size, v1 - step_size * current_grad)
        else: # If we don't : Line search ; to be implemented ; not working for now
            current_f_value = f.value(v1)
            while (True):
                v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step_size, v1 - step_size * current_grad)
                if (f.value(v2) <= current_f_value + np.dot(np.transpose(current_grad), v2 - v1) + 1/2/step_size * np.linalg.norm(v1 - v2)**2):
                    break
                else:
                    step_size = step_size * 0.8
        v1 = v2     
        reg_path[:,i] = v1
        
    return reg_path
    
def accelerated_prox_descent (p, f, g, stop, step_size=-1, eta=default_eta,ini=0):
    if (ini==0):
        v2 = np.transpose(np.matrix(np.zeros(p)))
        v1 = np.transpose(np.matrix(np.zeros(p)))
    else:
        v2 = ini
        v1 = np.transpose(np.matrix(np.zeros(p)))
    
    reg_path = np.matrix(np.zeros((p, stop)))
    if (step_size < 0):
        step_size = 1000
    for i in range (stop):
        y = v2 + float(i)/float(i+3) * (v2 - v1)
        print np.sum(np.abs((v2 - v1)))
        v1 = v2
        
        current_grad = f.grad(y)
        if (step_size >= 0): # If we know the lipschitz constante
            v2 = (1 - eta(i)) * y + eta(i) * g.prox(step_size, y - step_size * current_grad)
        else: # If we don't : Line search ; to be implemented ; not working for now
            current_f_value = f.value(y)
            while (True):
                v2 = (1 - eta(i)) * y + eta(i) * g.prox(step_size, y - step_size * current_grad)
                if (f.value(v2) <= current_f_value + np.dot(np.transpose(current_grad), v2 - y) + 1/2/step_size * np.linalg.norm(y - v2)**2):
                    break
                else:
                    step_size = step_size * 0.8
        
        reg_path[:,i] = v2

    return reg_path