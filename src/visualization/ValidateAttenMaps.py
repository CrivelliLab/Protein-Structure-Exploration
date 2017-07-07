'''
ValidateAttenMaps.py
Updated: 06/27/17

2ce2-442 h
1aa9-360 h
1p2u-374 m
1n4q-173 h
2rga-402

README:

'''
from Visualizations import *
import os, sys
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

#- Global Variables
pdb_id = '1nvu'
rot_id = 233
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
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[1], c2d[0]]

    return array_3d

def apply_rotation(pdb_data, rotation):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    return rotated_pdb_data

if __name__ == '__main__':
    # File Paths
    path_to_project = '../../'
    curve_2d = path_to_project + 'data/source/SFC/'+ curve_2d
    curve_3d = path_to_project + 'data/source/SFC/'+ curve_3d
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)
    pdb = pdb_id + '-' + str(rot_id) + '.png'


    img = misc.imread('../../data/valid/attenmaps/' + pdb)
    img = img.astype('float')/255.0
    #img[img < 0.2] = 0
    h = encode_2d_to_3d(img, curve_3d, curve_2d)

    img = misc.imread('../../data/final/RAS-MD512-HH/' + pdb)
    img = img.astype('float')/255.0
    x = img[:,:,0]
    x = encode_2d_to_3d(x, curve_3d, curve_2d)
    y = img[:,:,1]
    y = encode_2d_to_3d(y, curve_3d, curve_2d)
    z = img[:,:,2]
    z = encode_2d_to_3d(z, curve_3d, curve_2d)

    c = [x,y,z,h]

    #display_3d_array(c)

    entries = np.load('../../data/inter/RAS.npy')
    pdb_data = None
    rot = None
    for e in entries:
        if e[0] == pdb_id and e[1] == rot_id:
            pdb_data = e[2]
            rot = e[3]

    pdb_data = apply_rotation(pdb_data, rot)
    dia = 0
    for channel in pdb_data:
        temp = np.amax(np.abs(channel[:, 1:])) + 2
        if temp > dia: dia = temp

    display_3d_mesh(pdb_data, h, dia)
