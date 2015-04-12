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

# Smooth term
class Nuclear_Norm (prox_descent.Base_f_Function):
    def prox (self, mu, W):        
        U, sigma, V = np.linalg.svd(W)

        sigma = np.clip(sigma - mu, 0, np.inf)
        D1 = np.zeros((U.shape[1], V.shape[0]))
        np.fill_diagonal(D1, sigma)
             
        D1 = np.dot(U, np.dot(D1,V))
        
        return D1

# Operator that puts unknown positions in a matrix to 0
class Projection_On_Known_Positions:
    known_positions = 0
    
    # known positions = matrix (1 if the posiions is known, 0 otherwise)
    def __init__(self, known_positions):
        self.known_positions = known_positions
        
    def value (self, M):
        return np.multiply(self.known_positions, M)
        
    def transpose_value (self, M):
        return np.multiply(self.known_positions, M)
