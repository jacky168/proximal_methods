# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:37:34 2015

@author: Simon Matet

Functions for optimization of OLS with Tychonoff elastic net regularization
"""

import numpy as np
import prox_descent

# Elastic net : beta*|x|_1 + alpha/2 * |x|^2 #
class ElaNet_Function (prox_descent.Base_g_Function):
    alpha = 1
    beta = 1
    
    def __init__(self, beta, alpha):
      self.alpha = alpha
      self.beta = beta
      
    def prox (self , mu, v):
        return 1 / (1 + self.alpha*self.beta*mu) * (np.clip(v - self.beta*mu, 0, np.inf) - np.clip(-1*v - self.beta*mu, 0, np.inf))

class Square_Loss_Function (prox_descent.Base_f_Function):
    phi = 0
    Y = 0
    grad_lip_cte = -1
    
    def __init__ (self, phi, Y):
        self.phi = phi
        self.Y = Y
        
        
    def grad (self, v):
        return 2 * np.dot(np.transpose(self.phi), np.dot(self.phi, v) - self.Y)
        
    def value (self, v):
        return np.linalg.norm(np.dot(self.phi, v) - self.Y) **2
        
    def get_grad_lip_cte (self):
        return self.grad_lip_cte
