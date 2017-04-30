# -*- coding: utf-8 -*

'''
@author: PY131, created on 17.4.29
this is an implementation of RBF network
'''

class RBP_network: 
    '''
    the definition of BP network class
    '''
    
    def __init__(self):
        
        '''
        initial variables
        '''
        # node number each layer
        # input neuron number equals to input variables' number
        self.h_n = 0
        # output layer contains only one neuron
        
        # output value for each layer     
        self.b = []  # hidden layer
        self.y = 0.0 # output
        
        # parameters (w, b, c)
        self.w    = []  # weight of the link between hidden neuron and output neuron
        self.beta = []  # scale index of Gaussian-RBF
        self.c    = []  # center of Gaussian-RBF]
        
        # initial the learning rate
        self.lr = 0.05
        
    def CreateNN(self, nh, centers, learningrate):
        '''
        build a RBF network structure and initial parameters
        @param  nh : the neuron number of in layer
        @param centers: matrix [h_n * i_n] the center parameters object to hidden layer neurons
        @param learningrate: learning rate of gradient algorithm
        '''    
        # dependent packages
        import numpy as np       
               
        # assignment of hidden neuron number
        self.h_n = nh
        
        # initial value of output for each layer
        self.b = np.zeros(self.h_n)
        # self.y = 0.0
    
        # initial centers
        self.c = centers
    
        # initial weights for each link (random initialization)
        self.w    = np.zeros(self.h_n)
        self.beta = np.zeros(self.h_n)
        for h in range(self.h_n):  
            self.w[h]    = rand(0, 1)
            self.beta[h] = rand(0, 1)
    
        # initial learning rate
        self.lr = learningrate
    
    def Pred(self, x):
        '''
        predict process through the network
        @param x: array, input array for input layer
        @param y: float, output of the network 
        '''
        
        self.y = 0.0
        # activate hidden layer and calculating output
        for h in range(self.h_n):
            self.b[h] = RBF(x, self.beta[h], self.c[h])
            self.y += self.w[h] * self.b[h]

        return self.y
    
    def Batch_Pred(self, X):
        '''
        predict process through the network for batch data
        
        @param x: array, data set for input layer
        @param y: array, output of the networks
        '''
        
        y_pred = []
        # activate hidden layer and calculating output
        for i in range(len(X)):
            y_pred.append(self.Pred(X[i]))

        return y_pred

    def BackPropagateRBF(self, x, y):
        '''
        the implementation of special BP algorithm on one slide of sample for RBF network
        @param x, y: array and float, input and output of the data sample
        '''
        
        # dependent packages
        import numpy as np 

        # get current network output
        self.Pred(x)
        
        # calculate the gradient for hidden layer 
        g = np.zeros(self.h_n)
        for h in range(self.h_n):
            g[h] = (self.y - y) * self.b[h]    
        
        # updating the parameter
        for h in range(self.h_n):
            self.beta[h] += self.lr * g[h] * self.w[h] * np.linalg.norm(x-self.c[h],2)
            self.w[h] -= self.lr * g[h]
            

    def TrainRBF(self, data_in, data_out):
        '''
        BP training for RBF network
        @param data_in, data_out:
        @return e: accumulated error
        @return e_k: error array based on each step
        '''    
        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            self.BackPropagateRBF(x, y)
            
            # error in train set for each step
            y_delta2 = (self.y - y)**2  
            e_k.append(y_delta2/2)

        # total error of training
        e = sum(e_k)/len(e_k)
        
        return e, e_k
    
    
def RBF(x, beta, c):
    '''
    the definition of radial basis function (RBF)
    @param x: array, input variable
    @param beta: float, scale index
    @param c: array. center 
    '''
    
    # dependent packages
    from numpy.linalg import norm
    
    return norm(x-c,2)
        
def rand(a, b):
    '''
    the definition of random function
    @param a,b: the upper and lower limitation of the random value
    '''
    
    # dependent packages
    from random import random
    
    return (b - a) * random() + a   
  