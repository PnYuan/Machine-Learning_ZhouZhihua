#-*- coding: utf-8 -*

'''
@author: PY131, created on 17.4.29
this is an test of RBF network on xor io samples
'''

'''
preparation of data
'''
import numpy as np

# train set
X_trn = np.random.randint(0,2,(100,2))
y_trn = np.logical_xor(X_trn[:,0],X_trn[:,1])
# test set
X_tst = np.random.randint(0,2,(100,2))
y_tst = np.logical_xor(X_tst[:,0],X_tst[:,1])

'''
implementation of RBF network
'''
from RBF_BP import *

# generate the centers (4 centers with 2 dimensions) based on XOR data
centers = np.array([[0,0],[0,1],[1,0],[1,1]])

# construct the network
rbf_nn = RBP_network()  # initial a BP network class
rbf_nn.CreateNN(4, centers, learningrate=0.05)  # build the network structure

# parameter training
e = []
for i in range(10): 
    err, err_k = rbf_nn.TrainRBF(X_trn, y_trn)
    e.append(err)
    
# draw the convergence curve of output error by each step of iteration during training
import matplotlib.pyplot as plt 
f1 = plt.figure(1) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve")
plt.plot(e)
plt.show()

'''
model testing 
'''
y_pred = rbf_nn.Batch_Pred(X_tst);
count = 0
for i in range(len(y_pred)):
    if y_pred[i] >= 0.5 : y_pred[i] = True
    else : y_pred[i] = False
    if y_pred[i] == y_tst[i] : count += 1 
    
tst_err = 1 - count/len(y_tst)
print("test error rate: %.3f" % tst_err)







