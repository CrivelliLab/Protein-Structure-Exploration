'''
SpaceFillingCurves.py
Author: Rafael Zamora
Last Updated: 5/5/2017

This script is used to run visualizations of space filling curve transpositions.

'''

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from HilbertCurves import gen_hilbert_2D, gen_hilbert_3D

hilbert_3d = gen_hilbert_3D(8)
hilbert_2d = gen_hilbert_2D(12)

def gen_3d_dummy(x, y, z):
    '''
    Method generates 3-dimensional dummy data for the purpose of viewing how
    space filling curves transpose spatial information.

    '''
    dummy_3d = [[[(k + (j * x) + (i * y * x)) for k in range(z)] for j in range(y)] for i in range(x)]
    dummy_3d =  np.array(dummy_3d)
    dummy_3d = dummy_3d / np.max(dummy_3d)
    return dummy_3d

def spacefilling_3d_to_1d(array_3d, curve):
    '''
    Method applies hilbert curve to 3-d array.
    Returns a 1-d array.

    '''
    # Dimension Reduction Using Space FIlling Curves
    array_1d = np.zeros([len(curve),])
    for i in range(len(curve)):
        array_1d[i] = array_3d[curve[i][0], curve[i][1], curve[i][2]]

    return array_1d

def spacefilling_1d_to_2d(array_1d, curve):
    '''
    Method reconstructs 1-d array into 2-d array using hilbert curve.
    Returns a 2-d array.

    '''
    s = int(np.sqrt(len(curve)))
    array_2d = np.zeros([s,s])
    for i in range(len(array_1d)):
        array_2d[curve[i][0], curve[i][1]] = array_1d[i]
    return array_2d

def display_3d_array(array_3d, mask=None, curve=None):
    '''
    Method displays 3d array.

    '''
    # Generate Coordinates
    shape = array_3d.shape
    coords = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                coords.append([i,j,k])
    coords = np.array(coords)

    # Flatten Array
    flat_array = array_3d.flatten()

    # Apply Mask
    if mask:
        temp_flat = []
        temp_coords = []
        for i in range(len(flat_array)):
            if flat_array[i] > mask[0] and flat_array[i] < mask[1]:
                temp_flat.append(flat_array[i])
                temp_coords.append(coords[i])
        coords = np.array(temp_coords)
        flat_array = np.array(temp_flat)

    # Display 3D Plot
    colmap = cm.ScalarMappable(cmap='inferno')
    colmap.set_array(flat_array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if curve is not None: ax.plot(curve[:,0], curve[:,1], curve[:,2])
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=cm.inferno(flat_array), marker='o')
    fig.colorbar(colmap)
    plt.show()

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    plt.figure()
    plt.imshow(array_2d, interpolation="nearest", cmap='inferno')
    #plt.show()

if __name__ == '__main__':
    # Generate 3D Array
    x = 256
    y = 256
    z = 256
    dummy_3d = gen_3d_dummy(x, y, z)

    # Transpose 3D Array into 2D
    dummy_1d = spacefilling_3d_to_1d(dummy_3d, hilbert_3d)
    dummy_2d = spacefilling_1d_to_2d(dummy_1d, hilbert_2d)

    # Display Arrays
    display_2d_array(dummy_2d)
    plt.show()
    #display_3d_array(dummy_3d)
