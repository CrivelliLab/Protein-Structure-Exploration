'''
VisualizationTools.py
Last Updated: 5/5/2017

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def display_3d_array(array_3d, mask=None, curve=None):
    '''
    Method displays 3d array.

    '''

    xx, yy, zz = np.where(array_3d > 1)

    mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)

    mayavi.mlab.show()

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    plt.figure()
    plt.imshow(array_2d, interpolation="nearest", cmap='inferno')
    plt.show()

if __name__ == '__main__':
    for i in range(64):
        x = pow(2,i)
        sq = np.sqrt(x)
        cb = np.cbrt(x)
        if sq %1.0 == 0.0 and cb %1.0 == 0.0:
            print("\nHilbert 2D Curve:", int(sq), 'x', int(sq), ', order-', np.log2(sq))
            print("Hilbert 3D Curve:", int(cb), 'x', int(cb), 'x', int(cb), ', order-', np.log2(cb))
            print("Total Number of Pixels:", x)
            print("Estimated File Size for 1 bit Depth:", x/float(pow(2,30)))
