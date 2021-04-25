# used for manipulating directory paths
import os
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
data = genfromtxt('/Users/tarekashraf/Downloads/house_price.csv',delimiter=',')
X, y = data[1:10000, 1:], data[1:10000, 0]
Xcv, ycv = data[10001:14000, 1:], data[10001:14000, 0]
Xt, yt = data[14001:18000, 1:], data[14001:18000, 0]

def plotData(x, y):
    pyplot.plot(x, y, 'ro', ms=10, mec='k')

def  featureNormalize(X):
  X_norm = X.copy()
  mu = np.zeros(X.shape[1])
  sigma = np.zeros(X.shape[1])
  for i in range(18):
    mu = np.mean(X[:,i], axis = 0)
    sigma = np.std(X[:,i], axis = 0)
    X[:,i] = (X[:,i]-mu)/sigma
    Xcv[:,i] = (Xcv[:,i]-mu)/sigma
    Xt[:,i] = (Xt[:,i]-mu)/sigma
  
  return 
featureNormalize(X)

b = y.size
g = np.ones(b)
X = np.column_stack([X,g])
X[:,[0, 18]] = X[:,[18, 0]]
b = ycv.size
g = np.ones(b)
Xcv = np.column_stack([Xcv,g])
Xcv[:,[0, 18]] = Xcv[:,[18, 0]]
b = yt.size
g = np.ones(b)
Xt = np.column_stack([Xt,g])
Xt[:,[0, 18]] = Xt[:,[18, 0]]


def computeCostMulti(X, y, theta):

    m = y.shape[0] 
    J = 0
    E = X.dot(theta)
    J = 1/(2*m)*np.sum(np.square(E-y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters, lam):

    m = y.shape[0] 
    theta = theta.copy()
    J_history = []
    
    for i in range(num_iters):

        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X) + lam*theta
        J_history.append(computeCostMulti(Xcv, ycv, theta))
    
    return theta, J_history

def error(X, y, theta):
    m = y.shape[0] 
    J = 0
    E = X.dot(theta)
    J = 1/(2*m)*np.sum(np.square(E-y))
    return J

alpha = 0.01
num_iters = 100

theta = np.zeros(19)
theta, J_history = gradientDescentMulti(X, y, theta, alpha , num_iters, 0.02)
cf = error(Xt,yt,theta)
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')





