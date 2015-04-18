# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:05:07 2015

@author: Simon Matet

"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def get_low_rank_image (rank = 10):
    # load image
    X = mpimg.imread('lotus.jpg')
    X = X/255.0
    Black = np.sum(X, axis=2)
    
    
    UR, sigmaR, VR = np.linalg.svd(Black)
    sigmaR[rank:] = 0
    DR = np.zeros((UR.shape[1], VR.shape[0]))
    np.fill_diagonal(DR, sigmaR)
    DR = np.dot(UR, np.dot(DR,VR))

    return DR
    
def get_black_and_white_image ():
    X = mpimg.imread('lotus.jpg')
    X = X/255.0
    black = np.sum(X, axis=2)
    return black