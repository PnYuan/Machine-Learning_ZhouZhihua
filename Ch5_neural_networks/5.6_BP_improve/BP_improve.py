#-*- coding: utf-8 -*
    
'''''
@author: PY131
'''''
from sklearn.decomposition.tests.test_nmf import random_state

'''
preparation of data
'''
import pandas as pd
import matplotlib.pyplot as plt 

# online loading
from urllib.request import urlopen
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
raw_data = urlopen(url)     # download the file
attr = ['sepal_length','sepal_width','petal_length','petal_width','species'] 
dataset = pd.read_csv(raw_data, delimiter=",", header = None, names = attr)

# visualization of data
# import seaborn as sns
# sns.pairplot(dataset, hue='species', vars = ['sepal_length','petal_length']) 
# sns.plt.show()

# generation of input, output, label

# input variables (assignment directly)
X = dataset.iloc[:,:4].get_values()                 

# label (generation after transform output to categorical variables)
dataset.iloc[:,-1] = dataset.iloc[:,-1].astype('category')
label = dataset.iloc[:,4].values.categories      

# output 1 (generation after string categorical variables to numerical values)
dataset.iloc[:,4].cat.categories = [0,1,2]
y = dataset.iloc[:,4].get_values()

# output 2 (generation after one hot encoding)
Y = pd.get_dummies(dataset.iloc[:,4]).get_values()

'''
split of train set and test set (using sklearn function)
'''
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y, train_Y, test_Y = train_test_split(X,y,Y,test_size = 0.5, random_state = 42)  

'''
construction of BP network
'''
from BP_network import *
bpn1 = BP_network()  # initial a BP network class
bpn1.CreateNN(4, 5, 3, actfun = 'Sigmoid', learningrate = 0.05)  # build the network

'''
experiment of fixed learning rate
'''

# parameter training with fixed learning rate initial above
e = []
for i in range(1000): 
    err, err_k = bpn1.TrainStandard(train_X, train_Y)
    e.append(err)
    
# draw the convergence curve of output error by each step of iteration
import matplotlib.pyplot as plt 
f1 = plt.figure(1) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with fixed learning rate")
# plt.title("training error convergence curve\n learning rate = 0.05")
plt.plot(e)
# plt.show()

# get the test error in test set
pred = bpn1.PredLabel(test_X);
count  = 0
for i in range(len(test_y)) :
    if pred[i] == test_y[i]: count += 1

test_err = 1 - count/len(test_y)
print("test error rate: %.3f" % test_err)


'''
experiment of dynamic learning rate
'''

bpn2 = BP_network()  # initial a BP network class
bpn2.CreateNN(4, 5, 3, actfun = 'Sigmoid', learningrate = 0.05)  # build the network

# parameter training with fixed learning rate initial above
e = []
for i in range(1000): 
    err, err_k = bpn2.TrainStandard_Dynamic_Lr(train_X, train_Y)
    e.append(err)
    
# draw the convergence curve of output error by each step of iteration
# import matplotlib.pyplot as plt 
f2 = plt.figure(2) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with dynamic learning rate")
plt.plot(e)
# plt.show()

# get the test error in test set
pred = bpn2.PredLabel(test_X);
count  = 0
for i in range(len(test_y)) :
    if pred[i] == test_y[i]: count += 1

test_err = 1 - count/len(test_y)
print("test error rate: %.3f" % test_err)

plt.show()

print('haha')

