# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:11:31 2015

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import Functions.denoising as denoising
import Algos.reg_prox_descent as reg_prox_descent
import Utils.test_images as test_images


noise_level = 0.5
mu = 1.0
alpha = 1.0
stop = 100
step_size = alpha/16


clean_image = test_images.get_black_and_white_image()
#plt.imshow(clean_image,cmap = cm.Greys_r)
n,p = clean_image.shape

noise = np.random.normal(scale=noise_level, size = clean_image.shape)
noisy_image = clean_image + noise

z = np.zeros((noisy_image.shape[0] *3, noisy_image.shape[1]))
z[:noisy_image.shape[0], :noisy_image.shape[1]] = noisy_image

operator = denoising.operator()
f = denoising.isotropic(mu)
g = denoising.Scalar_Product (z)
comparator = denoising.comparator(clean_image)

x, score = reg_prox_descent.reg_prox_descent((3*n, p), (3*n, p), f, g, operator, alpha, 
                                      stop, step_size, z, comparator = comparator, verbose=True, 
                                      mode=1)
                             
    
plt.plot(range(stop), score, 'b')
plt.plot(range(stop), np.linalg.norm(clean_image - noisy_image) * np.ones(stop), 'r', linewidth = 3)
plt.imshow(x[:n,:],cmap = cm.Greys_r)