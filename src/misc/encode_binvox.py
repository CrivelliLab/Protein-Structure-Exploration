'''
encode_shapenet.py
Updated: 08/28/17

README:

'''
import os
from shutil import copyfile
import numpy as np
from scipy import misc, ndimage
from binvox import read_binvox, write_binvox
from mpi4py import MPI

#- Global Variables
curve_3d = 'hilbert_3d_6.npy'
curve_2d = 'hilbert_2d_9.npy'
data_folder = '../../data/raw/ShapeNetCore_v1/'
resized_folder = '../../data/raw/ShapeNetCore64/'
encoded_folder = '../../data/raw/EncodedShapeNetCore64/'

#- debug Settings
debug = False

################################################################################

def map_3d_to_2d(array_3d, curve_3d, curve_2d):
    '''
    Method maps 3D PDB array into 2D array.

    '''
    # Dimension Reduction Using Space Filling Curves from 3D to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

    return array_2d

def pad_32_to_64(array_3d):
    '''
    Method adds padding to 32x32x32 array to make it 64x64x64.

    '''
    padded_3d = np.zeros((64,64,64))
    for i in range(32):
        for j in range(32):
            for k in range(32):
                x = array_3d[i, j, k]
                padded_3d[(i*2),(j*2), (k*2)] = x
                padded_3d[(i*2),(j*2), (k*2)+1] = x
                padded_3d[(i*2),(j*2)+1, (k*2)] = x
                padded_3d[(i*2),(j*2)+1, (k*2)+1] = x
                padded_3d[(i*2)+1,(j*2), (k*2)] = x
                padded_3d[(i*2)+1,(j*2), (k*2)+1] = x
                padded_3d[(i*2)+1,(j*2)+1, (k*2)] = x
                padded_3d[(i*2)+1,(j*2)+1, (k*2)+1] = x
    return padded_3d

def reduce_128_to_64(array_3d):
    '''
    Method reduces 128x128x128 array to 64x64x64 array.

    '''
    reduce_3d = np.zeros((64,64,64))
    for i in range(64):
        for j in range(64):
            for k in range(64):
                x = np.max(array_3d[(i*2):(i*2)+2, (j*2):(j*2)+2, (k*2):(k*2)+2])
                reduce_3d[i,j,k] = x
    return reduce_3d

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    curve_2d = '../../data/misc/'+ curve_2d
    curve_3d = '../../data/misc/'+ curve_3d

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        if not os.path.exists(encoded_folder): os.makedirs(encoded_folder)
        if not os.path.exists(resized_folder): os.makedirs(resized_folder)
        entries = []
        for folder in sorted(os.listdir(data_folder)):
            if not os.path.isdir(data_folder+folder): continue
            if not os.path.exists(encoded_folder+folder): os.makedirs(encoded_folder+folder)
            if not os.path.exists(resized_folder+folder): os.makedirs(resized_folder+folder)
            for file_ in sorted(os.listdir(data_folder+folder)):
                if not file_endswith('.binvox'): continue
                binvox_path = data_folder+folder+'/'+ file_
                encoded_file_path = encoded_folder + folder +'/'+ file_[:-7] + '.png'
                #file_path = resized_folder + folder +'/'+ file_ + '.binvox'
                entries.append([binvox_path, encoded_file_path])
        entries = np.array(entries)
        np.random.shuffle(entries)
    else:
        entries = None

    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]

    print(len(entries))

    for i in range(len(entries)):
        entry = entries[i]

        # Load Binvox File
        model = read_binvox(entry[0])
        #model = reduce_128_to_64(model.astype('int'))
        encoded_2d = map_3d_to_2d(model, curve_3d, curve_2d)

        # Encode and Save Resulting Data Structure
        misc.imsave(entry[1], encoded_2d)
        #write_binvox(entry[2], model.astype(np.bool))
