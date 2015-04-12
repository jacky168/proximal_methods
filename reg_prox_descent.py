# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:16:20 2015

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:42:58 2015

@author: Simon Matet

"""

import numpy as np
        
def default_eta(k):
    return 1  
       
       
"""  Proximal Descent Method for dual problem :
     Minimizes dual of f(x) + g(linear_operator(x)) + epsilon/2 * ||x||^2; f convex differentiable ; g convex
     Arguments :
         - p          : integer : number of covariates
         - f function : object with method prox and (and maybe value of dual)
                 • Value is optionnal if grad_lip_cte is non negative.
         - g function : object with method prox (dual of g in the above problem)
                 • Should inherit Base_g_Function
         - linear_operator : object with method apply
         - epsilon : epsilon
         - stop      : integer : number of iterations
         - step size : default = -1 for unknown (does a line search, need f.value)
         - (optional) eta : averaging factor ; default = 1
         - (optional) ini : initialiation value ; default = 0
         
"""

# Only works with epsilon = 1
def reg_prox_descent (dim_primal, dim_dual, f, g, linear_operator, epsilon, stop, step_size, z, eta=default_eta,ini=np.zeros(10)):
    if (ini.all()==0):
        v1 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
    else:
        v1 = ini
            
    reg_path = np.zeros((dim_primal[0], dim_primal[1], stop))
    diff = np.zeros(stop)
    
    
    for i in range (stop):
        if (i%5 == 0):
            print i
        reg_path[:,:,i] = f.prox(1/epsilon, z - linear_operator.transpose_value(v1)/epsilon)
        current_grad = -1 * linear_operator.value(reg_path[:,:,i])        
        v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step_size, v1 - step_size * current_grad)
        diff[i] = np.linalg.norm(v2 - v1)
        v1 = v2     
        
    return reg_path, diff
    
    
def accelerated_reg_prox_descent (dim_primal, dim_dual, f, g, linear_operator, epsilon, stop, step_size, z, eta=default_eta,ini=np.zeros(10)):
    if (ini.all()==0):
        v1 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
        v2 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
    else:
        v1 = ini
        v2 = ini
            
    reg_path = np.zeros((dim_primal[0], dim_primal[1], stop))      
    diff = np.zeros(stop)
    
    for i in range (stop):
        if (i%5 == 0):
            print i
        y = v2 + float(i)/float(i+3) * (v2 - v1)
        diff[i] = np.linalg.norm(v2 - v1)
        v1 = v2
        reg_path[:,:,i] = f.prox(1/epsilon, z - linear_operator.transpose_value(v1)/epsilon)
        current_grad = -1 * linear_operator.value(reg_path[:,:,i])
        
        v2 = (1 - eta(i)) * y + eta(i) * g.prox(step_size, y - step_size * current_grad)
        
        
    return reg_path, diff