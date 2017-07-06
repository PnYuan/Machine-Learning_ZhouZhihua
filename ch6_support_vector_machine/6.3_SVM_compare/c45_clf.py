# coding = <utf-8>

'''
@author: PY131
'''

data_file_train     = "../data/btrain.csv"
data_file_valid     = "../data/bvalidate.csv"
data_file_test      = "../data/btest.csv"
data_file_datatype  = "../data/datatypes.csv"

# C4.5 for classification on breast_cancer data set

from sklearn.datasets import load_breast_cancer
import pandas as pd

##### generation of data set -> .csv file
data_set = load_breast_cancer()

X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

from sklearn.cross_validation import train_test_split

# generation of train set and testing set
X_train, X_test, y_train, y_test   = train_test_split(X,       y,       test_size=0.5, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5, random_state=0)

df_train = pd.DataFrame(X_train)
df_train.columns = feature_names
df_train['class'] = y_train
df_train.to_csv(data_file_train)

df_valid = pd.DataFrame(X_valid)
df_valid.columns = feature_names
df_valid['class'] = y_valid
df_valid.to_csv(data_file_valid)

df_test = pd.DataFrame(X_test)
df_test.columns = feature_names
df_test['class'] = y_test
df_test.to_csv(data_file_test)

##### learning the tree and testing
# add command line in debug/run arguments as: https://github.com/ryanmadden/decision-tree
import decision_tree
decision_tree.main()

# the result -> results.csv
from sklearn import metrics
df_result = pd.read_csv(open('results.csv', 'r'))
y_pred = df_result['class'].values
accuracy = metrics.accuracy_score(y_test, y_pred)  
print('accuracy of C4.5 tree: %.3f' % accuracy)

print(' - PY131 -')

