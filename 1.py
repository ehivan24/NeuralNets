'''
Created on Jun 12, 2015

@author: edwingsantos
'''
'''
import numpy as np

alist = [5,6,7,8,9]
arr = np.array(alist)

arr = np.zeros(5)


arr = np.logspace(0, 1, 100, base=10.0)

image = np.zeros((5,5)).astype(int) + 255
image2 = np.ones((5,5)).astype(int) + 255

print arr, '\n', image, '\n', image2


arrid = np.arange(1000)
arr3d = np.reshape(arrid,(10,10,10))

print arr3d


alist = [[34,6,7],[23,4,5]]
arr = np.array(alist)

print arr[:,2]
print arr[0,:]

arr = np.arange(5)
index = np.where(arr >= 2)
print index

A = np.matrix([[4,5,-6],
               [-5,-8,1],
               [-1,5,9]
               ])
B = np.matrix([[12],
               [-1],
               [10]
               ])

X = A ** (-1) *B
print X

'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b):
    return a*x+b

x = np.linspace(0,10,100)
y =func(x, 1, 2)

yp = y+0.9*np.random.normal(size=len(x))

popt, pcov =curve_fit(func, x, yp)
print (popt)

plt.plot(yp)
plt.show()

