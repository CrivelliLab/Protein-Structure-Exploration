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
    plt.show()
