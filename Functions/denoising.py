# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:42:12 2015

@author: user
"""

import numpy as np
import copy
import Algos.prox_descent as prox_descent

"""
Primal variable is 
[[Image]]
[[Vertical variations]]
[[Horizontal variatios]]

Dual variable is
[[Image]]
[[Divergence1]]
[[Divergence2]]
"""

# "Not smooth" term
# In practice, Y_pratice = (noisy_image, 0)
class Scalar_Product (prox_descent.Base_g_Function):
    Y = 0
    
    # This allows us to set parameters easily
    def __init__(self, Y):
      self.Y = Y
      
    def prox (self , mu, v):
        return v - mu*self.Y

# Smooth term 
# May be a bit long (square roots...)
class isotropic (prox_descent.Base_f_Function):
    mu = 1
    
    def prox (self, gamma, W):
        n = W.shape[0]/3
        proj = np.zeros(W.shape)
        divide = np.clip(np.sqrt(np.square(W[n:2*n,:]) + np.square(W[2*n:3*n,:])) * (gamma*self.mu), 1, np.inf)       
        proj[n:2*n,:] = W[n:2*n,:] / divide
        proj[2*n:3*n,:] = W[2*n:3*n,:] / divide
        return W - proj
        
    def __init__(self, mu):
            self.mu = mu
        
    # Not implemented (we know the lip constante)
    def moreau_env_dual_value (self, w, alpha):
        return -1
 
 # Anisotropic versio (not that the other one is really isotropic)
class anisotropic (prox_descent.Base_f_Function):
    mu = 1
    
    def prox (self, gamma, W):
        n = W.shape[0]/3
        mon_prox = copy.deepcopy(W)
        mon_prox[n:,:] = np.clip(mon_prox[n:,:] - self.mu*gamma, 0, np.inf) + np.clip(mon_prox[n:,:]+self.mu*gamma, -1*np.inf, 0)
        return mon_prox
        
    def __init__(self, mu):
            self.mu = mu
        
    # Not implemented (we know the lip constante)
    def moreau_env_dual_value (self, w, alpha):
        return -1
 
# (  Id , 0 )
# (-Grad, Id)       
class operator():
    def value (self, M):
        valeur = copy.deepcopy(M)
        n = M.shape[0]/3
        p = M.shape[1]
        result = valeur
        result[n:2*n -1,:] = result[n:2*n -1,:] -valeur[1:n,:] + valeur[0:n-1,:]
        result[2*n:,:p-1] = result[2*n:,:p-1] -valeur[:n,1:p] + valeur[:n,0:p-1]
                
        
        return result
        
    # (U,V) -> (U + div V, V)
    def transpose_value (self, W):
        n = W.shape[0]/3
        p = W.shape[1]
        valeur = copy.deepcopy(W)
        result = valeur
        result[:n-1,:] = result[:n-1,:] + valeur[n:2*n-1,:]
        result[1:n,:] = result[1:n,:] - valeur[n:2*n-1,:]
        result[:n,:p-1] = result[:n,:p-1] + valeur[2*n:,:p-1]
        result[:n,1:p] = result[:n,1:p] - valeur[2*n:,:p-1]       
        
        return result
   
     
class comparator():
    ref = 0
    
    def score (self, x):
        return np.linalg.norm(x[:x.shape[0]/3, :] - self.ref)
        
    def __init__ (self, noiseless_image):
        self.ref = noiseless_image