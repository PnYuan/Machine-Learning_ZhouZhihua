#--coding:utf-8--

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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from time import time
import sompy.sompy

mapsize = [20,20]
som = sompy.SOMFactory.build(X_train ,mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch')
som.train(n_job=1, shared_memory=True, verbose='info') 


v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
# could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
v .show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
# v.save('2d_packed_test')



som.component_names = ['1','2']
v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6) #which_dim='all' default

print('haha')



