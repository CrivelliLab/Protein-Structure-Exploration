'''
EncodePDB.py
Author: Rafael Zamora
Updated: 6/16/17

'''
import os, sys, time
import numpy as np
import itertools as it

# MPI
from mpi4py import MPI

# Space Filling Curves
from ZCurves import *

# PDB Proccessing
from ProcessingTools import *

# Image Saving
from scipy import misc

# Global Variables
folder = None
dest_folder = None
dynamic_bounding = True
sample_dim = 64
range_ = [-100, 100]
sel_channels = ['hydrophobic', 'polar', 'charged']

# Verbose Settings
debug = True
visualize = True
debug_pdb = '1aa9.pdb.gz'

# Visualization Tools
if visualize: from VisualizationTools import *

# Defined Rotations
axis_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
theta_list = [(np.pi*i)/4  for i in range(8)]

if __name__ == '__main__':

    # Directory Args
    args = sys.argv[1:]
    if len(args) >= 2:
        folder = args[0]
        dest_folder = args[1]
        if folder[-1] != '/': folder += '/'
        if dest_folder[-1] != '/': dest_folder += '/'
    elif len(args) == 1:
        folder = args[0]
        if folder[-1] != '/': folder += '/'
    else:
        print("Argument Error...")
        exit()

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()
    if debug:
        print("MPI Info...")
        print "Rank:", rank
        print "Number of Cores:", cores

    # Root Node Data Init
    if rank == 0:
        # Read PDBs
        pdb_files = []
        for line in sorted(os.listdir(folder)): pdb_files.append(line)
        if debug:
            print "Encoding PDBs in:", folder
            print "Total PDB Entries:", len(pdb_files)

        # Generate Space Filling Curves
        if debug: print("Generating 3D Curve...")
        curve_3d = gen_zcurve_3D(pow(sample_dim, 3))
        if debug: print("Generating 2D Curve...")
        curve_2d = gen_zcurve_2D(pow(sample_dim, 3))

        # Generate Rotations
        if debug: print("Generating Rotations...")
        base_rotations = []
        for axis in axis_list:
            base_rotations.append([])
            for theta in theta_list:
                rotation = rotation_matrix(axis, theta)
                base_rotations[-1].append(rotation)
        base_indices = [[i for i in range(len(base_rotations[j]))] for j in range(len(base_rotations))]
        indices = list(it.product(*base_indices))
        rotations = []
        for index in indices:
            comb_rotation = []
            for i in range(len(base_rotations)):
                comb_rotation.append(base_rotations[i][index[i]])
            rotations.append(comb_rotation)
        rotations = np.array(rotations)

        # Generate Entries
        entries = []
        for i in range(len(pdb_files)):
            for j in range(len(rotations)):
                entries.append([pdb_files[i], rotations[j], j])
        entries = np.array(entries, dtype=object)

    else:
        curve_3d = None
        curve_2d = None
        entries = None

    # MPI Broadcast Data
    if debug: print("Broadcasting Data To Nodes...")
    curve_3d = comm.bcast(curve_3d, root=0)
    curve_2d = comm.bcast(curve_2d, root=0)
    entries = comm.bcast(entries, root=0)

    # MPI Cut Entries Per Node
    entries = np.array_split(entries, cores)[rank]
    if debug:
        print "MPI Core", rank
        print "Processing", len(entries), "Entries"

    # Process Rotations
    for i in range(len(entries)):
        pdb_file = entries[i][0]
        if debug:
            start = time.time()
            print('Processing Entry ' + str(i) + '...')
            if debug_pdb: pdb_file = debug_pdb
        rot = entries[i][1]
        rot_id = entries[i][2]
        pdb_data, dia = get_pdb_data(folder + pdb_file, channels=sel_channels, rot=rot, debug=debug)
        dia += 2

        # Process Channels
        encoded_pdb_2d = []
        pdb_3d_model = []
        for i in range(len(pdb_data)):
            if debug: print('Processing Channel ' + str(i) + '...')
            pdb_data_res = pdb_data[i]
            if pdb_data_res is None: continue

            # Generate PDB Channel 3D Voxel Model
            if dynamic_bounding:
                bounds = [pow(-1,i+1)*dia for i in range(6)]
                #pdb_3d_res = dep_gen_3d_model(pdb_data_res, dia, -dia, sample_dim)
                pdb_3d_res = gen_3d_pdb(pdb_data_res, bounds, sample_dim, debug=debug)
            else:
                bounds = range_ + range_ + range_
                pdb_3d_res = gen_3d_pdb(pdb_data_res, bounds, sample_dim, debug=debug)
            pdb_3d_model.append(pdb_3d_res)

            # Encode 3D Model with Space Filling Curve
            encoded_res_2d = encode_3d_pdb(pdb_3d_res, curve_3d, curve_2d, debug=debug)
            encoded_pdb_2d.append(encoded_res_2d)

        # Transpose Array
        encoded_pdb_2d = np.array(encoded_pdb_2d)
        encoded_pdb_2d = np.transpose(encoded_pdb_2d, (2,1,0))

        # Save Encoded PDB to Numpy Array File.
        if debug: print("Saving Encoded PDB...")
        if dest_folder:
            file_path = dest_folder + pdb_file.split('.')[0] + '_'+ str(rot_id) +'.png'
        else:
            foldername = folder.split('/')[-2]
            file_path = folder[:-len(foldername)-1]+'Processed-'+ foldername +'/'+ pdb_file.split('.')[0] + '_'+ str(rot_id) +'.png'
        #np.savez_compressed(file_path, a=encoded_pdb_2d, b=rot)
        misc.imsave(file_path, encoded_pdb_2d)

        if debug: print "Processed in: ", time.time() - start, ' sec'

        # Visualize PDB Data
        if visualize:
            print('Visualizing All Channels...')
            #display_3d_points(pdb_data)
            #display_3d_mesh(pdb_data)
            display_3d_array(pdb_3d_model)
            display_2d_array(encoded_pdb_2d)

        if debug: exit()
