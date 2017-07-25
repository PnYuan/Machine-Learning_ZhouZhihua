# coding = <utf-8>

'''
@author: PY131
'''

# input file
data_file_watermelon_3a = "../data/watermelon_3a.csv"

########## data loading and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR  # for SVR model
from sklearn.model_selection import GridSearchCV  # for optimal parameter search

# loading data
df = pd.read_csv(data_file_watermelon_3a, header=None, )
df.columns = ['id', 'density', 'sugar_content', 'label']
df.set_index(['id'])
X = df[['density']].values
y = df[['sugar_content']].values

# generate model and fitting
# linear
# svr = GridSearchCV(SVR(kernel='linear'), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3]})

# rbf
# svr = GridSearchCV(SVR(kernel='rbf'), cv=5, param_grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]})

# ploy

svr = GridSearchCV(SVR(kernel='poly'), cv=5, param_grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]})

svr.fit(X, y)

sv_ratio = svr.best_estimator_.support_.shape[0] / len(X)
print("Support vector ratio: %.3f" % sv_ratio)

y_svr = svr.predict(X)

sv_ind = svr.best_estimator_.support_
plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors', zorder=2)
plt.scatter(X, y, c='k', label='data', zorder=1)
plt.plot(X, y_svr, c='orange', label='SVR fit curve with poly kernel')
plt.xlabel('density')
plt.ylabel('sugar_ratio')
plt.title('SVR on watermelon3.0a')
plt.legend()
plt.show()


print(' - PY131 - ')