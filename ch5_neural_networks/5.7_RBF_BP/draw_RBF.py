# -*- coding: utf-8 -*

def DrawRBF(beta, c):
    '''
    draw a figure of RBF in 3D space
    
    @param beta, scale index in RBF
    @param c: the center of RBF, here c is required to be 2D
    '''        
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
      
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    
    # Gaussian RBF
    Z = np.exp( -beta*((X-c[0])**2+(Y-c[1])**2) )
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
#         linewidth=0, antialiased=False)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#     fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    import numpy as np
    c = np.zeros(2)
    DrawRBF(0.2,c)
    
    
    