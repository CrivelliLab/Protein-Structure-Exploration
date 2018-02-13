'''
display_2d.py
Updated: 2/12/18

Script is used to visualize 2D representations of data.

'''
import os
import numpy as np
from matplotlib.cm import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Data Path
path = "../../data/KrasHras/Hras/1aa9_A"

################################################################################

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    Param:
        array_2d - np.array
        attenmap - np.array

    '''
    # Display 2D Plot
    n = array_2d.shape[-1]
    cm = [brg(float(i)/(n-1))[:3] for i in range(n)]
    for i in range(n):
        if i == 0: cmap = ListedColormap([[0,0,0,0.0], cm[i][:3]])
        else: cmap = ListedColormap([[0,0,0,0], cm[i][:3]])
        plt.imshow(array_2d[:,:,i], cmap=cmap, interpolation='nearest')
    plt.show()

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load Array
    array_2d = np.load(path + '/' + path.split('/')[-1] + '-2d.npz')['arr_0'].astype('int')

    # Display 2D Array
    display_2d_array(array_2d)
