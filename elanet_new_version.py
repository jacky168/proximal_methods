# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:21:09 2015

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:42:57 2015

@author: user
"""

import numpy as np
import prox_descent

# "Not smooth" term
class Scalar_Product (prox_descent.Base_g_Function):
    Y = 0
    
    # This allows us to set parameters easily
    def __init__(self, Y):
      self.Y = Y
      
    def prox (self , mu, v):
        return v - mu*self.Y

# Smooth term \mu * |.|_1
class L1_Norm (prox_descent.Base_f_Function):
    mu = 1
    
    def prox (self, gamma, W):        
        return np.clip(W - self.mu*gamma, 0, np.inf) + np.clip(W+self.mu*gamma, -1*np.inf, 0)
        
    def __init__(self, mu):
            self.mu = mu
            
    def dual_value (self, w):
        return -1 # Not implemented
        
    def moreau_env_dual_value (self, w, alpha):
        return 1/(2*alpha) * np.linalg.norm(np.clip(np.abs(w) - self.mu, 0, np.inf))**2
        
# Operator that puts unknown positions in a matrix to 0
class mult:
    Phi = 0
    
    # known positions = matrix (1 if the posiions is known, 0 otherwise)
    def __init__(self, Phi):
        self.Phi = Phi
        
    def value (self, M):
        return np.dot(self.Phi, M)
        
    def transpose_value (self, M):
        return np.dot(np.transpose(self.Phi), M)