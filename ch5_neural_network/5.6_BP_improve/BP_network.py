#-*- coding: utf-8 -*
    
'''''
@author: PY131, created on 17.4.24
this is an implementation of BP network
'''''

'''
the definition of BP network class
'''
class BP_network: 

    def __init__(self):
        
        '''
        initial variables
        '''
        # node number each layer
        self.i_n = 0           
        self.h_n = 0   
        self.o_n = 0

        # output value for each layer
        self.i_v = []       
        self.h_v = []
        self.o_v = []

        # parameters (w, t)
        self.ih_w = []    # weight for each link
        self.ho_w = []
        self.h_t  = []    # threshold for each neuron
        self.o_t  = []

        # definition of alternative activation functions and it's derivation
        self.fun = {
            'Sigmoid': Sigmoid, 
            'SigmoidDerivate': SigmoidDerivate,
            'Tanh': Tanh, 
            'TanhDerivate': TanhDerivate,
            
            # for more, add here
            }
        
        # initial the learning rate
        self.lr1 = []  # output layer
        self.lr2 = []  # hidden layer
        
        
    def CreateNN(self, ni, nh, no, actfun, learningrate):
        '''
        build a BP network structure and initial parameters
        @param ni, nh, no: the neuron number of each layer
        @param actfun: string, the name of activation function
        @param learningrate: learning rate of gradient algorithm
        '''
        
        # dependent packages
        import numpy as np       
               
        # assignment of node number
        self.i_n = ni
        self.h_n = nh
        self.o_n = no
        
        # initial value of output for each layer
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(self.o_n)

        # initial weights for each link (random initialization)
        self.ih_w = np.zeros([self.i_n, self.h_n])
        self.ho_w = np.zeros([self.h_n, self.o_n])
        for i in range(self.i_n):  
            for h in range(self.h_n): 
                self.ih_w[i][h] = rand(0, 1)
        for h in range(self.h_n):  
            for j in range(self.o_n): 
                self.ho_w[h][j] = rand(0, 1)
                
        # initial threshold for each neuron
        self.h_t = np.zeros(self.h_n)
        self.o_t = np.zeros(self.o_n)
        for h in range(self.h_n): self.h_t[h] = rand(0, 1)
        for j in range(self.o_n): self.o_t[j] = rand(0, 1)

        # initial activation function
        self.af  = self.fun[actfun]
        self.afd = self.fun[actfun+'Derivate']

        # initial learning rate
        self.lr1 = np.ones(self.o_n) * learningrate
        self.lr2 = np.ones(self.h_n) * learningrate

    def Pred(self, x):
        '''
        predict process through the network
        @param x: the input array for input layer
        '''
        
        # activate input layer
        for i in range(self.i_n):
            self.i_v[i] = x[i]
            
        # activate hidden layer
        for h in range(self.h_n):
            total = 0.0
            for i in range(self.i_n):
                total += self.i_v[i] * self.ih_w[i][h]
            self.h_v[h] = self.af(total - self.h_t[h])
            
        # activate output layer
        for j in range(self.o_n):
            total = 0.0
            for h in range(self.h_n):
                total += self.h_v[h] * self.ho_w[h][j]
            self.o_v[j] = self.af(total - self.o_t[j])
        
    '''
    for fixed learning rate
    '''    
        
    def BackPropagate(self, x, y):
        '''
        the implementation of BP algorithms on one slide of sample
        
        @param x, y: array, input and output of the data sample
        '''
        
        # dependent packages
        import numpy as np 

        # get current network output
        self.Pred(x)
        
        # calculate the gradient based on output
        o_grid = np.zeros(self.o_n) 
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j]) * self.afd(self.o_v[j])
        
        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * self.afd(self.h_v[h])   

        # updating the parameter
        for h in range(self.h_n):  
            for j in range(self.o_n): 
                self.ho_w[h][j] += self.lr1[j] * o_grid[j] * self.h_v[h]
           
        for i in range(self.i_n):  
            for h in range(self.h_n): 
                self.ih_w[i][h] += self.lr2[h] * h_grid[h] * self.i_v[i]     

        for j in range(self.o_n):
            self.o_t[j] -= self.lr1[j] * o_grid[j]    
                
        for h in range(self.h_n):
            self.h_t[h] -= self.lr2[h] * h_grid[h]
   
   
    def TrainStandard(self, data_in, data_out):
        '''
        standard BP training
        @param lr, learning rate, default 0.05
        @return: e, accumulated error
        @return: e_k, error array of each step
        '''    
        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            self.BackPropagate(x, y)
            
            # error in train set for each step
            y_delta2 = 0.0
            for j in range(self.o_n):
                y_delta2 += (self.o_v[j] - y[j]) * (self.o_v[j] - y[j])  
            e_k.append(y_delta2/2)

        # total error of training
        e = sum(e_k)/len(e_k)
        
        return e, e_k
    
            
    '''
    for dynamic learning rate
    '''   
    
    def BackPropagate_Dynamic_Lr(self, x, y, d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p, alpha):
        '''
        the implementation of BP algorithms on one slide of sample
        
        @param x, y: array, input and output of the data sample
        @param d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p: adjust values (delta) of last step
        @param o_grid_p, h_grid_p: gradient of last step
        @param alpha: forget factor
        
        @return adjust values (delta) of ho_w, ih_w, o_t, h_t, 
                and gradient value of o_grid, h_grid for this step
        '''
        
        # dependent packages
        import numpy as np 

        # get current network output
        self.Pred(x)
        
        # calculate the gradient based on output
        o_grid = np.zeros(self.o_n) 
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j]) * self.afd(self.o_v[j])
        
        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * self.afd(self.h_v[h])   

        # updating the parameter
        lamda = np.sign(o_grid * o_grid_p)
        o_grid_p = o_grid
        for h in range(self.h_n):  
            for j in range(self.o_n): 
                # adjust learning rate
                o_grid_p[j] = o_grid[j]
                lr = self.lr1[j] * ( 3 ** lamda[j] )                 
                self.lr1[j] = 0.5 if lr > 0.5 else (0.005 if lr < 0.005 else lr)
                # updating parameter
                d_ho_w_p[h][j] = self.lr1[j] * o_grid[j] * self.h_v[h] + alpha * d_ho_w_p[h][j]
                self.ho_w[h][j] += d_ho_w_p[h][j]
           
        lamda = np.sign(h_grid * h_grid_p)
        h_grid_p = h_grid
        for i in range(self.i_n):  
            for h in range(self.h_n):    
                # adjust learning rate
                lr = self.lr2[h] * ( 3 ** lamda[h] )
                self.lr2[j] = 0.5 if lr > 0.5 else (0.005 if lr < 0.005 else lr)
                
                # updating parameter
                d_ih_w_p[i][h] = self.lr2[h] * h_grid[h] * self.i_v[i] + alpha * d_ih_w_p[i][h]
                self.ih_w[i][h] += d_ih_w_p[i][h]

        for j in range(self.o_n):
            d_o_t_p[j] = -( self.lr1[j] * o_grid[j] + alpha * d_o_t_p[j] )
            self.o_t[j] += d_o_t_p[j]
                
        for h in range(self.h_n):
            d_h_t_p[h] = -( self.lr2[h] * h_grid[h] + alpha * d_h_t_p[h] )
            self.h_t[h] += d_h_t_p[h]
            
        return d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p
            
    
    def TrainStandard_Dynamic_Lr(self, data_in, data_out):
        '''
        standard BP training
        @param lr, learning rate, default 0.05
        @return: e, accumulated error
        @return: e_k, error array of each step
        '''
        # dependent packages
        import numpy as np 
        
        d_ih_w_p = np.zeros([self.i_n, self.h_n])  # initial delta values = 0.0
        d_ho_w_p = np.zeros([self.h_n, self.o_n])
        d_h_t_p  = np.zeros(self.h_n)
        d_o_t_p  = np.zeros(self.o_n)
        
        o_grid_p = np.zeros(self.o_n)  # initial gradient = 0.01
        h_grid_p = np.zeros(self.h_n)      

        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p \
                = self.BackPropagate_Dynamic_Lr(x, y, d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, 
                                                o_grid_p, h_grid_p, 0.2)
                                    
            # error in train set for each step
            y_delta2 = 0.0
            for j in range(self.o_n):
                y_delta2 += (self.o_v[j] - y[j]) * (self.o_v[j] - y[j])  
            e_k.append(y_delta2/2)

        # total error of training
        e = sum(e_k)/len(e_k)
        
        return e, e_k
        
    def PredLabel(self, X):
        '''
        predict process through the network
        
        @param X: the input sample set for input layer
        @return: y, array, output set (0,1,2... - class) based on [winner-takes-all] 
        '''    
        import numpy as np
               
        y = []
        
        for m in range(len(X)):
            self.Pred(X[m])
#             if self.o_v[0] > 0.5:  y.append(1)
#             else : y.append(0)
            max_y = self.o_v[0]
            label = 0
            for j in range(1,self.o_n):
                if max_y < self.o_v[j]: 
                    label = j
                    max_y = self.o_v[j]
            y.append(label)
           
        return np.array(y)     
            

'''
the definition of activation functions
'''
def Sigmoid(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import exp
    return 1.0 / (1.0 + exp(-x))
def SigmoidDerivate(y):
    return y * (1 - y)

def Tanh(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import tanh
    return tanh(x)
def TanhDerivate(y):
    return 1 - y*y

'''
the definition of random function
'''
def rand(a, b):
    '''
    random value generation for parameter initialization
    @param a,b: the upper and lower limitation of the random value
    '''
    from random import random
    return (b - a) * random() + a

    
