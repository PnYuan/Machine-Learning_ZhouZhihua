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

som = SimpleSOMMapper((10, 1), 1000, learning_rate=0.05)      
som.train(X)

# print(som.K)  # see the center value of Kohonen layer node
centers = som.K.reshape([10,2])
f2 = plt.figure()  
plt.scatter(centers[:,0], centers[:,1], s=1000)
plt.title('Kohonen layer unit center')  

# the result of mapping on watermelon_dataset
mapped = som(X) # mapping result of X via SOM
f3 = plt.figure(3) 
mapped = mapped[:,0].reshape([1, len(mapped[:,0])])
plt.imshow(mapped, origin='lower', cmap = 'gray')

mapped = som(X)

plt.title('watermelon_3a mapping graph')
for i, m in enumerate(mapped):
    plt.text(i, 0, y[i], ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.3, lw=0))
plt.show()

