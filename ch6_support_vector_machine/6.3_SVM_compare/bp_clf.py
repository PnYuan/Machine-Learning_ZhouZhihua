# coding = <utf-8>

'''
@author: PY131
'''

# BP net for classification on breast_cancer data set

##### loading data
from sklearn.datasets import load_breast_cancer
data_set = load_breast_cancer()

X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

# data normalization
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

# construction of data in pybrain's formation
from pybrain.datasets import ClassificationDataSet
ds = ClassificationDataSet(30, 1, nb_classes=2, class_labels=y)  
for i in range(len(y)): 
    ds.appendLinked(X[i], y[i])
ds.calculateStatistics()

# split of training and testing dataset
tstdata_temp, trndata_temp = ds.splitWithProportion(0.5)  
tstdata = ClassificationDataSet(30, 1, nb_classes=2)
for n in range(0, tstdata_temp.getLength()):
    tstdata.appendLinked( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(30, 1, nb_classes=2)
for n in range(0, trndata_temp.getLength()):
    trndata.appendLinked( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

##### build net and training
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError

n_hidden = 500
bp_nn = buildNetwork(trndata.indim, n_hidden, trndata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(bp_nn, 
                          dataset=trndata,
                          verbose=True,
                          momentum=0.5,
                          learningrate=0.0001,
                          batchlearning=True)
err_train, err_valid = trainer.trainUntilConvergence(maxEpochs=1000, 
                                                     validationProportion=0.25)

# convergence curve for accumulative BP algorithm process
import matplotlib.pyplot as plt
f1 = plt.figure(1)
plt.plot(err_train,'b',err_valid,'r')
plt.title('BP network classification')  
plt.ylabel('error rate')  
plt.xlabel('epochs')  
plt.show()

# testing
tst_result = percentError(trainer.testOnClassData(tstdata),tstdata['class'] )
print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tst_result)

print(' - PY131 -')
