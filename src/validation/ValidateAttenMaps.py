'''
ValidateAttenMaps.py
Updated: 06/27/17

README:

The following script is used to run validation experiments on various 3D to 2D
spacefilling curve transpositions.

Global variables used to run experiments are defined under #- Global Variables.

'''
from mayavi import mlab
import os, sys
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm

#- Global Variables
curve_3d = 'hilbert_3D6.npy'
curve_2d = 'hilbert_2D9.npy'

#- Verbose Settings
debug = True

################################################################################

def load_atten_maps():
    '''
    '''
    # Read PDB IMG File Names
    atten_imgs = []
    for line in sorted(os.listdir('../../data/valid/attenmaps/')):
        if line.endswith('.png'): atten_imgs.append(line)
    atten_imgs = np.array(atten_imgs)

    # Load PDB Images
    atten_maps = []
    for j in tqdm(range(len(atten_imgs))):
        img = misc.imread('../../data/valid/attenmaps/'+atten_imgs[j])
        img = img.astype('float')/255.0
        atten_maps.append(img)

    return atten_maps

def display_3d_array(array_3d):
    '''
    Method displays 3d array.

    '''
    # Dislay 3D Voxel Rendering
    for i in range(len(array_3d)):
        if i == 1: c = (1, 0, 0)
        elif i == 2: c = (0, 1, 0)
        elif i == 0: c = (0, 0, 1)
        else: c = (1,1,1)
        xx, yy, zz = np.where(array_3d[i] >= 0.5)
        mlab.points3d(xx, yy, zz, mode="cube", color=c, scale_factor=1)
    mlab.show()

def encode_2d_to_3d(array_2d, curve_3d, curve_2d):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''

    # Dimension Reduction Using Space Filling Curves to 2D
    s = int(np.cbrt(len(curve_3d)))
    array_3d = np.zeros([s,s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[0], c2d[1]]

    return array_3d

if __name__ == '__main__':
    # File Paths
    path_to_project = '../../'
    curve_2d = path_to_project + 'data/source/SFC/'+ curve_2d
    curve_3d = path_to_project + 'data/source/SFC/'+ curve_3d
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    img = misc.imread('../../data/valid/attenmaps/1aa9-360.png')
    img = img.astype('float')/255.0
    img[img < 0.5] = 0
    plt.imshow(img)
    plt.show()
    h = encode_2d_to_3d(img, curve_3d, curve_2d)

    img = misc.imread('../../data/final/RAS-SD512-HH/1aa9-360.png')
    img = img.astype('float')/255.0
    x = img[:,:,0]
    x = encode_2d_to_3d(x, curve_3d, curve_2d)
    y = img[:,:,1]
    y = encode_2d_to_3d(y, curve_3d, curve_2d)
    z = img[:,:,2]
    z = encode_2d_to_3d(z, curve_3d, curve_2d)

    c = [x,y,z,h]
    display_3d_array(c)
