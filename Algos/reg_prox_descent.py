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
        
def constant(k):
    return 1  
    
stop_threshold = 1.5
       
       
"""  Proximal Descent Method for dual problem :
     Minimizes dual of f(x) + g(linear_operator(x)) + epsilon/2 * ||x - z||^2; f convex differentiable ; g convex
     Arguments :
         - p          : integer : number of covariates
         - f function : object with method prox and (and maybe value of dual)
                 • Value is optionnal if grad_lip_cte is non negative.
         - g function : object with method prox (dual of g in the above problem)
                 • Should inherit Base_g_Function
         - linear_operator : object with method apply
         - epsilon : epsilon
         - stop      : integer : number of iterations
         - step size : set -1 for unknown (does a line search, needs f.moreau_env_dual_value)
         - z
         - (opional) comparator : object with method score : 
                                     compares to reference parameters (for benchmarking)
                                     (default : constant)
         - (optional) eta : averaging factor ; default = 1
         - (optional) ini : initialiation value ; default = 0
         - (optional) verbose : default False. To see progress
         - (optional) mode : 0 : normal : returns list of best parameters and list of scores
                             1 : amnesic : returns list of scores and last set of values
         returs regularization path (list of best parameters)
"""


def reg_prox_descent (dim_primal, dim_dual, f, g, linear_operator, epsilon, 
                      stop, step_size, z, comparator=constant, eta=constant,ini=np.zeros(10),
                      verbose=False, mode=0):
    if (ini.all()==0):
        v1 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
    else:
        v1 = ini
            
    if (mode == 0):
        reg_path = np.zeros((dim_primal[0], dim_primal[1], stop))
        
    score = np.zeros(stop)
    x = v1
    
    backtracking = False
    if (step_size < 0):
        backtracking = True
        backtracking_multiplier = .5
        step_size = 1.0
    
    minimum = np.inf
    
    
    for i in range (stop):
        if (i%5 == 0  and verbose):
            print i
        x = f.prox(1/epsilon, z - linear_operator.transpose_value(v1)/epsilon)
        current_grad = -1 * linear_operator.value(x) 
        if (mode == 0):
                reg_path[:,:,i] = x
        score[i] = comparator.score(x)
        
        if (score[i] < minimum):
            minimum = score[i]           
        if ((score[i] - minimum)/minimum > stop_threshold):
            break

        if (not backtracking):
            v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step_size, v1 - step_size * current_grad)
            
        else:
            # Could be optimized to remove one call to a prox-like method
            ftilde = f.moreau_env_dual_value(z - linear_operator.transpose_value(v1), epsilon)
            
            while(True):
                v2 = (1 - eta(i)) * v1 + eta(i) * g.prox(step_size, v1 - step_size * current_grad)
                if (f.moreau_env_dual_value(z - linear_operator.transpose_value(v2), epsilon) <= \
                        ftilde \
                        + np.sum(np.multiply(v2-v1, current_grad)) \
                        + 1/(2 * step_size) * np.linalg.norm(v2 - v1)**2):
                    break
                else:
                    step_size = step_size * backtracking_multiplier
                
        v1 = v2

        
    if (mode == 0):
        return reg_path, score
    else:
        return x, score
    
    
"""
Same with acceleration
"""
def accelerated_reg_prox_descent (dim_primal, dim_dual, f, g, linear_operator, 
                                  epsilon, stop, step_size, z, comparator=constant,
                                  eta=constant, ini=np.zeros(10), verbose = False, mode=0):
    if (ini.all()==0):
        v1 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
        v2 = np.matrix(np.zeros((dim_dual[0], dim_dual[1])))
    else:
        v1 = ini
        v2 = ini
    
    y = v1
    
    if (mode == 0):
        reg_path = np.zeros((dim_primal[0], dim_primal[1], stop))
    score = np.zeros(stop)
    x = v1
    
    backtracking = False
    if (step_size < 0):
        backtracking = True
        backtracking_multiplier = .5
        step_size = 1.0
    t1 = 1.0
    minimum = np.inf
    
    for i in range (stop):
        if (not backtracking):
            if (i%5 == 0  and verbose):
                print i            
            # Little approximation which may or may not be right. To check.
            x = f.prox(1/epsilon, z - linear_operator.transpose_value(y)/epsilon)
            if (mode == 0):
                reg_path[:,:,i] = x
            score[i] = comparator.score(x)
            current_grad = -1 * linear_operator.value(x)     
            v2 = (1 - eta(i)) * y + eta(i) * g.prox(step_size, y - step_size * current_grad)
            
        else:
            x = f.prox(1/epsilon, z - linear_operator.transpose_value(y)/epsilon)
            if (mode == 0):
                reg_path[:,:,i] = x
            score[i] = comparator.score(x)
            current_grad = -1 * linear_operator.value(x)
            # Could be optimized to remove one call to a prox-like method
            ftilde = f.moreau_env_dual_value(z - linear_operator.transpose_value(y), epsilon)
            
            while(True):
                v2 = (1 - eta(i)) * y + eta(i) * g.prox(step_size, y - step_size * current_grad)
                if (f.moreau_env_dual_value(z - linear_operator.transpose_value(v2), epsilon) <= \
                        ftilde \
                        + np.sum(np.multiply(v2-y, current_grad)) \
                        + 1.0/(2.0 * step_size) * np.linalg.norm(v2 - y)**2):
                    break
                else:
                    step_size = step_size * backtracking_multiplier
                    
        t2 = (1.0 + np.sqrt(1.0 + 4.0*t1**2.0))/2.0
        y = v2 + (t1 - 1.0) / t2 * (v2 - v1) 
        t1 = t2
        v1 = v2
        
        if (score[i] < minimum):
            minimum = score[i]           
        if ((score[i] - minimum)/minimum > stop_threshold):
            break
        
    if (mode == 0):
        return reg_path, score
    else:
        return x, score