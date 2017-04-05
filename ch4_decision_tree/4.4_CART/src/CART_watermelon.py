# -*- coding: utf-8 -*

'''''
create on 2017/3/24, the day after our national football team beat south korea
@author: PY131
'''''

'''
import data and pre-analysis through data visualization
'''
# using pandas dataframe for .csv read which contains chinese char.
import pandas as pd
data_file_encode = "gb18030"
with open("../data/watermelon_2.csv", mode = 'r', encoding = data_file_encode) as data_file:
    df = pd.read_csv(data_file)

'''
implementation of CART rely on decision_tree.py
'''
import decision_tree

# dicision tree visualization using pydotplus.graphviz
index_train = [0,1,2,5,6,9,13,14,15,16]

df_train = df.iloc[index_train]
df_test  = df.drop(index_train)

# generate a full tree
root = decision_tree.TreeGenerate(df_train)
decision_tree.DrawPNG(root, "decision_tree_full.png")
print("accuracy of full tree: %.3f" % decision_tree.PredictAccuracy(root, df_test))

# pre-purning
root = decision_tree.PrePurn(df_train, df_test)
decision_tree.DrawPNG(root, "decision_tree_pre.png")
print("accuracy of pre-purning tree: %.3f" % decision_tree.PredictAccuracy(root, df_test))

# # post-puring
root = decision_tree.TreeGenerate(df_train)
decision_tree.PostPurn(root, df_test)
decision_tree.DrawPNG(root, "decision_tree_post.png")
print("accuracy of post-purning tree: %.3f" % decision_tree.PredictAccuracy(root, df_test))


# print the accuracy
# k-folds cross prediction
accuracy_scores = []
n = len(df.index)
k = 5
for i in range(k):
    m = int(n/k)
    test = []
    for j in range(i*m, i*m+m):
        test.append(j)
        
    df_train = df.drop(test)
    df_test = df.iloc[test]
    root = decision_tree.TreeGenerate(df_train)  # generate the tree
    decision_tree.PostPurn(root, df_test)  # post-purning
    
    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.Predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1
            
    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy) 

# print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ", end = "")
for i in range(k):
    print("%.3f  " % accuracy_scores[i], end = "")
    accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f" % (accuracy_sum/k))


 
    