""" Script generating a signal from a function """
""" Simon Matet - 27.03.2015 """

import numpy as np
import matplotlib.pyplot as plt

""" Function to regress. """
def default_f (X):
    return np.cos(np.pi*X) + 1.5*np.sin(3*np.pi*X) - 2*X + 3*np.exp(X)

def generate (n, p, sigma_noise=1, res=10000, f=default_f):
    """ Generates the kernel (cos in the first columns, sin after) """
    points = np.transpose((np.matrix(range(n)) - float(n-1)/2 )/float(n-1) *2) # Type-1 data
    Xbuff = np.dot(points, np.matrix(np.ones(p)))
    for k in range(p):
        Xbuff[:,k] = Xbuff[:,k] * k# * np.pi    
    X = np.matrix(np.zeros((n,2*p)))
    X[:,:p] = np.cos(Xbuff)
    X[:,p:2*p] = np.sin(Xbuff)
    
    
    """ Generates the observations """
    Y = f(points) + np.transpose(np.matrix(np.random.normal(loc=0.0, scale=sigma_noise, size=n)))
    plt.plot(points, Y, 'ro')
    
    """ Plots reference function (with higher resolution) """
    points_ref = np.transpose((np.matrix(range(res)) - float(res-1)/2 )/float(res-1) *2)
    Xbuffref = np.dot(points_ref, np.matrix(np.ones(p)))
    for k in range(p):
        Xbuffref[:,k] = Xbuffref[:,k] * k# * np.pi    
    X_ref = np.matrix(np.zeros((res,2*p)))
    X_ref[:,:p] = np.cos(Xbuffref)
    X_ref[:,p:2*p] = np.sin(Xbuffref)
    Y_ref = f(points_ref)
    
    return points, X, Y, points_ref, X_ref, Y_ref