# -*- coding: utf-8 -*

def Draw_Lateral_Inhibition(sigma, c):
    '''
    draw a figure of Lateral Inhibition in 3D space
    
    @param beta, scale index in RBF
    @param c: the center of RBF, here c is required to be 2D
    '''        

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['axes.unicode_minus'] = False # for minus displey
    
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
      
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    
    # T_j,J(x)
    Z = np.exp( -((X-c[0])**2+(Y-c[1])**2) / 2*sigma**2 )
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.contour(X, Y, Z, zdir='x', offset=-10, cmap=plt.cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=10,  cmap=plt.cm.coolwarm)
    
    ax.set_xlabel('X')
    ax.set_xlim(-10, 10)
    ax.set_ylabel('Y')
    ax.set_ylim(-10, 10)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 1)
    plt.title("lateral_inhibition")

    plt.show()

if __name__ == '__main__':
    import numpy as np
    c = np.zeros(2)
    Draw_Lateral_Inhibition(0.2,c)