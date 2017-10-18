'''
encode_objs.py
Updated: 09/25/17

README:

This script is used to encode 3D object files into 2D images. Objects are
voxelized using binvox into 256x256x256 array and then down-sampled to
64x64x64. The resulting array is encoded into 2D images using hilbert curves.

'''
import os
import trimesh
import numpy as np
from mpi4py import MPI
from shutil import move
from scipy import misc, ndimage
from binvox_io import read_binvox, write_binvox

#- Global Variables
curve_3d = 'hilbert_3d_6.npy'
curve_2d = 'hilbert_2d_9.npy'
data_folder = '../../../data/raw/ModelNet10/'
encoded_folder = '../../../data/raw/ModelNet10_rot30/'
binvox_folder = '../../../data/raw/ModelNet10_rot30_binvox/'
file_type = '.off'
nb_rots = 30
seed = 45

################################################################################

def binvox_off_file_256(off_file):
    '''
    '''
    cmd = "./binvox_linux -cb -e -d 256 " + off_file
    os.system(cmd)

def binvox_off_file_64(off_file):
    '''
    '''
    cmd = "./binvox_linux -cb -e -d 64 " + off_file
    os.system(cmd)

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

def reduce_256_to_64(array_3d):
    '''
    Method reduces 256x256x256 array to 64x64x64 array.

    '''
    reduce_3d = np.zeros((64,64,64))
    for i in range(64):
        for j in range(64):
            for k in range(64):
                x = np.sum(array_3d[(i*4):(i*4)+4, (j*4):(j*4)+4, (k*4):(k*4)+4])
                reduce_3d[i,j,k] = x/64.0
    return reduce_3d

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    curve_2d = 'data/'+ curve_2d
    curve_3d = 'data/'+ curve_3d

    # Load Curves
    print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    np.random.seed(seed)

    # Get file paths and broadcast
    if rank == 0:
        if not os.path.exists(encoded_folder): os.makedirs(encoded_folder)
        if not os.path.exists(binvox_folder): os.makedirs(binvox_folder)
        if not os.path.exists(encoded_folder+'train/'): os.makedirs(encoded_folder+'train/')
        if not os.path.exists(binvox_folder+'train/'): os.makedirs(binvox_folder+'train/')
        if not os.path.exists(encoded_folder+'test/'): os.makedirs(encoded_folder+'test/')
        if not os.path.exists(binvox_folder+'test/'): os.makedirs(binvox_folder+'test/')
        entries = []
        for folder in sorted(os.listdir(data_folder)):
            if not os.path.exists(encoded_folder+'train/'+folder): os.makedirs(encoded_folder+'train/'+folder)
            if not os.path.exists(binvox_folder+'train/'+folder): os.makedirs(binvox_folder+'train/'+folder)
            if not os.path.exists(encoded_folder+'test/'+folder): os.makedirs(encoded_folder+'test/'+folder)
            if not os.path.exists(binvox_folder+'test/'+folder): os.makedirs(binvox_folder+'test/'+folder)
            for file_ in sorted(os.listdir(data_folder+folder+'/train')):
                if not file_.endswith(file_type): continue
                obj_path = 'train/'+folder+'/'+ file_
                for i in range(nb_rots): entries.append([obj_path, i])
            for file_ in sorted(os.listdir(data_folder+folder+'/test')):
                if not file_.endswith(file_type): continue
                obj_path = 'test/'+folder+'/'+ file_
                for i in range(nb_rots): entries.append([obj_path, i])
        entries = np.array(entries)
        rand_seeds = [np.random.rand(3) for i in range(nb_rots)]
        np.random.shuffle(entries)
    else:
        entries = None
        rand_seeds = None
    entries = comm.bcast(entries, root=0)
    rand_seeds = comm.bcast(rand_seeds, root=0)
    entries = np.array_split(entries, cores)[rank]
    print(len(entries))

    for i in range(len(entries)):
        entry = entries[i][0]
        index = int(entries[i][1])
        inverse_entry = entry.split('/')
        inverse_entry = inverse_entry[1] + '/' + inverse_entry[0] + '/' + inverse_entry[2]
        mesh_base = trimesh.load_mesh(data_folder+inverse_entry)

        mesh_rr = mesh_base.apply_transform(trimesh.transformations.random_rotation_matrix(rand_seeds[index]))
        off_obj = trimesh.io.export.export_off(mesh_rr)
        inverse_mesh_rr_file = inverse_entry.split('.')[0] + '-r' + str(index) + '.off'
        mesh_rr_file = entry.split('.')[0] + '-r' + str(index) + '.off'
        with open(data_folder + inverse_mesh_rr_file, 'w') as fb: fb.write(off_obj)

        # Binvox object file
        binvox_off_file_256(data_folder + inverse_mesh_rr_file)

        # Load Binvox File
        binvox_file = inverse_mesh_rr_file.split('.')[0] + '.binvox'
        model = read_binvox(data_folder+binvox_file)

        # Encode and Save Resulting Data Structure
        model = reduce_256_to_64(model.astype('int'))
        encoded_2d = map_3d_to_2d(model, curve_3d, curve_2d)
        png_file = mesh_rr_file.split('.')[0] + '.png'
        misc.imsave(encoded_folder+png_file, encoded_2d)
        os.remove(data_folder+binvox_file)

        # Copy file to binvoxed folder
        binvox_off_file_64(data_folder + inverse_mesh_rr_file)
        move(data_folder+binvox_file, binvox_folder+mesh_rr_file.split('.')[0] + '.binvox')
        os.remove(data_folder + inverse_mesh_rr_file)
