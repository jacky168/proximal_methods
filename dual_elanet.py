# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:29:23 2015

@author: Simon Matet

Functions for the dual problem of OLS with elastic net penalization 
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
class ElaNet_Dual_With_Operator (prox_descent.Base_f_Function):
    phi = 0
    alpha = 0
    
    def __init__ (self, phi, alpha):
        self.phi = phi
        self.alpha = alpha
        if (alpha == 0):
            raise ZeroDivisionError("Argument 'alpha' should not be 0")

    # w is the dual variable.
    def grad (self, w):
        buff = np.dot(np.transpose(self.phi), w)
        buff = np.clip(buff -1, 0, np.inf) + np.clip(buff+1, -1*np.inf, 0)
        return np.dot(self.phi, buff) / self.alpha
        
    # w is the dual variable.
    def value (self, w):
        return np.sum(np.square(np.clip(np.abs(np.dot(np.transpose(self.phi), w)) -1, 0, np.inf))) /2 /self.alpha
