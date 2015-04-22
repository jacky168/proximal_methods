# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:11:31 2015

@author: user
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import Functions.denoising as denoising
import Algos.reg_prox_descent as reg_prox_descent
import Utils.test_images as test_images


noise_level = 0.1
mu = 1000.0
alpha = 0.001
stop = 25
step_size = alpha/25
salt_and_pepper_ratio = 0.25


clean_image = test_images.get_black_and_white_image()
#plt.imshow(clean_image,cmap = cm.Greys_r)
n,p = clean_image.shape

"""
# Gaussian Noise
noise = np.random.normal(scale=noise_level, size = clean_image.shape)
noisy_image = clean_image + noise
"""
# Salt and pepper noise
mask = np.random.binomial(1, 0.5, (n,p))
salt_and_pepper = np.random.binomial(1, salt_and_pepper_ratio, (n,p))
salt_and_pepper = np.multiply(salt_and_pepper, 1 - 2*mask)
noisy_image = copy.deepcopy(clean_image)
noisy_image[salt_and_pepper == 1] = 1
noisy_image[salt_and_pepper == -1] = 0


extended_image = np.zeros((noisy_image.shape[0] *3, noisy_image.shape[1]))
extended_image[:noisy_image.shape[0], :noisy_image.shape[1]] = noisy_image

operator = denoising.operator()
f = denoising.isotropic(mu)
g = denoising.Scalar_Product (extended_image)
comparator = denoising.comparator(clean_image)

z = np.zeros((noisy_image.shape[0] *3, noisy_image.shape[1]))
z[:noisy_image.shape[0], :noisy_image.shape[1]] = noisy_image

x, score = reg_prox_descent.accelerated_reg_prox_descent((3*n, p), (3*n, p), f, g, operator, alpha, 
                                      stop, step_size, z, comparator = comparator, verbose=True, 
                                      mode=0)
                        
    
plt.plot(range(stop), score, 'b')
plt.plot(range(stop), np.linalg.norm(clean_image - noisy_image) * np.ones(stop), 'r', linewidth = 3)
#plt.imshow(x[:n,:],cmap = cm.Greys_r)

def image():
    plt.imshow(x[:n,:],cmap = cm.Greys_r)
    
def image2(i):
    plt.imshow(x[:n,:,i],cmap = cm.Greys_r)

def plot():
    plt.plot(range(stop), score, 'b')
    plt.plot(range(stop), np.linalg.norm(clean_image - noisy_image) * np.ones(stop), 'r', linewidth = 3)
    
def noisy():
    plt.imshow(noisy_image,cmap = cm.Greys_r)

def clear():
    plt.imshow(clean_image,cmap = cm.Greys_r)