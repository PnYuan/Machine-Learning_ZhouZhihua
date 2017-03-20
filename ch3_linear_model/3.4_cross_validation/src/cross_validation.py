# -*- coding: utf-8 -*

'''''
Created on 2017/3/20   
   
@author: PY131
'''''

'''
1-st. iris data set importing and visualization using seaborn
'''
import numpy as np
import seaborn as sns
sns.set(style="white", color_codes=True)
iris = sns.load_dataset("iris")
X = iris.values[50:150,0:4]
y = iris.values[50:150,4]

# iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
# sns.pairplot(iris,hue='species') 
# sns.plt.show()

'''
2-nd logistic regression using sklearn
'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# log-regression lib model
log_model = LogisticRegression()
m = np.shape(X)[0]

# 10-folds CV
y_pred = cross_val_predict(log_model, X, y, cv=10)
print(metrics.accuracy_score(y, y_pred))
    
# LOOCV
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0;
for train, test in loo.split(X):
    log_model.fit(X[train], y[train])  # fitting
    y_p = log_model.predict(X[test])
    if y_p == y[test] : accuracy += 1  
print(accuracy / np.shape(X)[0])

# m = np.shape(X)[0]
# scores_loo = cross_val_score(log_model, X, y, cv=m)
# print(scores_loo)
# # prediction using 10-folds
# y_pred_loo = cross_val_predict(log_model, X, y, cv=m)
# print(metrics.accuracy_score(y, y_pred_loo))

'''
transfusion-blood dats set analysis
'''
# import numpy as np  # for matrix calculation
dataset_transfusion = np.loadtxt('../data/transfusion.data', delimiter=",", skiprows=1)
X2 = dataset_transfusion[:,0:4]
y2 = dataset_transfusion[:,4]

# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.model_selection import cross_val_predict

# log-regression lib model
log_model = LogisticRegression()
m = np.shape(X2)[0]

# 10-folds CV
y2_pred = cross_val_predict(log_model, X2, y2, cv=10)
print(metrics.accuracy_score(y2, y2_pred))
    
# LOOCV
# from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0;
for train, test in loo.split(X2):
    log_model.fit(X2[train], y2[train])  # fitting
    y2_p = log_model.predict(X2[test])
    if y2_p == y2[test] : accuracy += 1  
print(accuracy / np.shape(X2)[0])

print('haha')
