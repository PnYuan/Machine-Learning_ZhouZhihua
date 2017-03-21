# -*- coding: utf-8 -*

'''''
Create on 2017/3/21

@author PY131
'''''

import numpy as np

'''
get the projective point(2D) of a point to a line

@param point: the coordinate of the point form as [a,b]
@param line: the line parameters form as [k, t] which means y = k*x + t
@return: the coordinate of the projective point  
'''
def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if   k == 0:      return [a, t]
    elif k == np.inf: return [0, b]
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]

         
    
    
    