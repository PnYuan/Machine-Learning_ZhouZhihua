# coding = <utf-8>

'''
@author: PY131
'''

# SVM for classification on breast_cancer data set

##### loading data
from sklearn.datasets import load_breast_cancer
data_set = load_breast_cancer()

X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

# draw scatter
import matplotlib.pyplot as plt

f1 = plt.figure(1)  

p1 = plt.scatter(X[y==0,0], X[y==0,1], color='r', label=target_names[0])   # feature
p2 = plt.scatter(X[y==1,0], X[y==1,1], color='g', label=target_names[1])   # feature
plt.xlabel(feature_names[0])  
plt.ylabel(feature_names[1])  
plt.legend(loc='upper right')
plt.grid(True, linewidth=0.3)

plt.show()

# data normalization
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

##### model fitting and testing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
import numpy as np

# generation of train set and testing set
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.5, random_state=0)

# model fitting, testing, visualization
# based on linear kernel and rbf kernel 
for fig_num, kernel in enumerate(('linear', 'rbf')):
    accuracy = []
    c = []
    for C in range(1, 1000, 1):
        # initial
        clf = svm.SVC(C=C, kernel=kernel)
        # train
        clf.fit(X_train, y_train)
        # testing 
        y_pred = clf.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        c.append(C)
        
    print('max accuracy of %s kernel SVM: %.3f' % (kernel,max(accuracy)))
    
    # draw accuracy
    f2 = plt.figure(2)
    plt.plot(c, accuracy)
    plt.xlabel('penalty parameter')
    plt.ylabel('accuracy')
    
    plt.show()



print(' - PY131 -')
