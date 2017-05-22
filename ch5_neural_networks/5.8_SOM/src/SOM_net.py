# -*- coding: utf-8 -*

'''
@author: PY131, created on 2017.5.1
'''

'''
data loading and pre-processing
'''
import numpy as np  # for matrix calculation

# load the .csv file as a numpy matrix
data_file = open('../data/watermelon_3a.csv')
dataset = np.loadtxt(data_file, delimiter=",")

# separate the data from the target attributes
X = dataset[:,1:3]
y = dataset[:,3]    

# data normalization (z-scale) based on column
X_train = np.zeros(X.shape)
for j in range(X.shape[1]):
    for i in range(X.shape[0]):
        X_train[i,j] = (X[i,j] - X[:,j].mean())/X[:,j].std()
X = X_train

# draw scatter diagram to show original data
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.unicode_minus'] = False # for minus displey

f1 = plt.figure(1)       
plt.title('watermelon_3a')  
plt.xlabel('density')  
plt.ylabel('ratio_sugar')  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')  
# plt.show()

'''
implementation of SOM net based on <pymvpa2>
'''

# from mvpa2.suite import *
from mvpa2.suite import SimpleSOMMapper

som = SimpleSOMMapper((10, 1), 10000, learning_rate=0.05)      
som.train(X)

mapped = som(X)

# print(np.asarray(mapped).ndim) 
# print(np.asarray(mapped).shape[2]) # to check if is suited for imshow()

# b = np.zeros([som.K.shape[0], som.K.shape[1], som.K.shape[2]+1])
# for i in range(som.K.shape[0]):
#     for j in range(som.K.shape[1]):
#         b[i,j,0:som.K.shape[2]] = som.K[i,j,:]

f2 = plt.figure(2) 
plt.imshow(~mapped, origin='lower', cmap = 'gray')
mapped = som(X)

plt.title('watermelon_3a')
# for i, m in enumerate(mapped):
#     plt.text(m[0], m[1], y[i], ha='center', va='center',
#            bbox=dict(facecolor='white', alpha=0., lw=0))
plt.show()

