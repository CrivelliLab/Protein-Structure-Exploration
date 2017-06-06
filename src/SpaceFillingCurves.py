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

def gen_3d_dummy(x, y, z):
    '''
    Method generates 3-dimensional dummy data for the purpose of viewing how
    space filling curves transpose spatial information.

    '''
    dummy_3d = [[[(k + (j * x) + (i * y * x)) for k in range(z)] for j in range(y)] for i in range(x)]
    dummy_3d =  np.array(dummy_3d)
    dummy_3d = dummy_3d / np.max(dummy_3d)
    return dummy_3d

def hilbert_3d_to_1d(array_3d):
    '''
    Method applies hilbert curve to 3-d array.
    Returns a 1-d array.

    '''
    pass

def hilbert_1d_to_2d(array_1d):
    '''
    Method reconstructs 1-d array into 2-d array using hilbert curve.
    Returns a 2-d array.

    '''
    pass

def display_3d_array(array_3d, mask=None):
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
    plt.show()

if __name__ == '__main__':
    # Generate 3D Array
    x = 9
    y = 9
    z = 9
    dummy_3d = gen_3d_dummy(x, y, z)

    '''
    # Transpose 3D Array into 2D
    dummy_1d = hilbert_3d_to_1d(dummy_3d)
    dummy_2d = hilbert_1d_to_2(dummy_1d)

    '''
    dummy_2d = dummy_3d.flatten().reshape((27,27))

    # Display Arrays
    display_3d_array(dummy_3d)
    display_2d_array(dummy_2d)
