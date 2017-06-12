'''
VisualizationTools.py
Last Updated: 5/11/2017

'''
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

def display_3d_array(array_3d, mask=None, curve=None):
    '''
    Method displays 3d array.

    '''
    # Dislay 3D Rendering
    xx, yy, zz = np.where(array_3d >= 1)
    mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)
    mlab.show()

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    plt.figure()
    plt.imshow(array_2d, interpolation="nearest")
    plt.show()

def display_spacefilling_dim():
    '''
    Method displays dimensions of various order space filling curves.

    '''
    # Calculating Hilbert 3D to 2D Conversion Dimensions
    for i in range(64):
        x = pow(2,i)
        sq = np.sqrt(x)
        cb = np.cbrt(x)
        if sq %1.0 == 0.0 and cb %1.0 == 0.0:
            print "\nSpace Filling 2D Curve:", int(sq), 'x', int(sq), ', order-', np.log2(sq)
            print "Space Filling 3D Curve:", int(cb), 'x', int(cb), 'x', int(cb), ', order-', np.log2(cb)
            print "Total Number of Pixels:", x

def display_min_diameter_dist(min_diameters):
    '''
    '''
    plt.hist(min_diameters)
    plt.title("Minimum Diameters")
    plt.show()

if __name__ == '__main__':
    display_spacefilling_dim()
