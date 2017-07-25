查看相关答案和源代码，欢迎访问我的Github：[PY131/Machine-Learning_ZhouZhihua](https://github.com/PY131/Machine-Learning_ZhouZhihua).

## 6.2 支持向量分析实验 ##
> ![](Ch6/6.2.png)
> 
> ![](Ch6/6.2.data.png)

(注：本题实验基于python，另外，sklearn库已集成了libsvm库，并在其基础上扩展形成了自带svm工具库，这里我们采用该sklearn-svm工具库开展实验）

[查看本实验完整代码](https://github.com/PY131/Machine-Learning_ZhouZhihua/blob/master/ch6_support_vector_machine/6.2_SVM_test/sv_compare.py)

### 数据预处理 ###

生成数据```watermelon_3a.csv```，将类别编码为 0（否），1（是），基于pandas读取数据，做出可视化界面如下：

![](Ch6/6.2.scatter.png)

### 训练与分析 ###

采用```sklearn.svm.svc```训练并得出支持向量，实验段程序示意如下：
	
	```python
	from sklearn import svm 
    # initial
    svc = svm.SVC(C=1000, kernel=kernel)  # classifier 1 based on linear kernel
    # train
    svc.fit(X, y)
    # get support vectors
    sv = svc.support_vectors_
	```

绘制出决策边界，同时标记出支持向量如下图：

1. 线性核函数：

![](Ch6/6.2.linear.png)

2. 高斯核函数：

![](Ch6/6.2.rbf.png)

可以估计出，面向该题数据集，高斯核函数的拟合更好（间隔更小），且用到的支持向量更少（当前参数设置下有9个支持向量）。

### 参考 ###

本文涉及的一些参考资料如下：

 - sklearn官网 - [sklearn.svm.SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
 - sklearn官网 - [SVM Exercise（使用样例）](http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html#sphx-glr-auto-examples-exercises-plot-iris-exercise-py)