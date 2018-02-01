'''
generate_data.py
Updated: 1/10/19

This script is used to parallelize data generation using the data_generator object.
To generate a new dataset make sure to fetch pdbs first using the fetch_pdbs.py script.
Compressed numpy files for 3D, 2D, and 1D representations will be produced for each
PDB entry in dataset. Each PDB entry has their own designated folder which contains
numpy files as well as pdb file.

To run: $ python3 generate_data.py or $ mpirun -n N python3 generate_data.py

'''
import os
import scipy
import numpy as np
from mpi4py import MPI
from channels import *
from itertools import product
from data_generator import data_generator

# Data folder path
data_folder = '../../data/KrasHras/'

# Data generator parameters
size = 64               # Voxel matrix size ex. 64 -> 64**3 space
resolution = 1.0        # Resolution of unit voxel
thresh = 0.95           # percentage of protein which must be inside window
nb_rot = 1              # Number of random rotation augmentations
channels = [all_atoms]

################################################################################

all_chains = False
residue_indexes = None # Select only atoms of these indexes
seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    # MPI task distribution
    if rank == 0:
        tasks = []

        # Search for class directories
        for class_dir in sorted(os.listdir(data_folder)):
            if os.path.isdir(data_folder+class_dir):

                # Search for data directories
                for data_dir in sorted(os.listdir(data_folder+class_dir)):
                    if os.path.isdir(data_folder+class_dir+'/'+data_dir):

                        # Iterate over number of rotations
                        for j in range(nb_rot):
                            tasks.append([data_folder+class_dir+'/'+data_dir, j])

        # Shuffle for random distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Intialize Data Generator
    pdb_datagen = data_generator(size=size, resolution=resolution, thresh=thresh,
                                 nb_rots=nb_rot, channels=channels)

    # Generate data for each task
    for i in range(len(tasks)):

        # Parse task
        task = tasks[i][0].split('/')[-1].split('_')
        pdb_id = task[0]
        chain = task[1]
        rot = int(tasks[i][1])

        # Set path to pdb file
        pdb_path = tasks[i][0] + '/' + '_'.join(task) + '.pdb'

        # Generate and Save Data
        pdb_data = pdb_datagen.generate_data(pdb_path, chain, rot,
                                    res_i=residue_indexes, all_chains=all_chains)

        # if data was generated without error
        if len(pdb_data) > 0:
            print(pdb_id, chain, rot)

            # Get data for each dimensionality
            array_3d = pdb_data[0]
            array_2d = pdb_data[1]
            array_1d = pdb_data[2]

            if rot > 0:
                # If rotation augmentations set new save path
                path = tasks[i][0]
                task.append(str(rot))
                #if not os.path.exists(path): os.mkdir(path)

                # Save compressed numpy file for each representations
                np.savez(path + '/' + '_'.join(task) + '-1d.npz', array_1d.astype('bool'))
                np.savez(path + '/' + '_'.join(task) + '-2d.npz', array_2d.astype('bool'))
                np.savez(path + '/' + '_'.join(task) + '-3d.npz', array_3d.astype('bool'))

            else:
                # Save compressed numpy file for each representations
                np.savez(tasks[i][0] + '/' + '_'.join(task) + '-1d.npz', array_1d.astype('bool'))
                np.savez(tasks[i][0] + '/' + '_'.join(task) + '-2d.npz', array_2d.astype('bool'))
                np.savez(tasks[i][0] + '/' + '_'.join(task) + '-3d.npz', array_3d.astype('bool'))

            del array_3d, array_2d, array_1d

    print("Done")
