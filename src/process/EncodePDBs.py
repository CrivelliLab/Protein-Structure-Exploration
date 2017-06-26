'''
EncodePDB.py
Updated: 6/23/17

'''
import os, sys, time
import numpy as np
from scipy import misc, ndimage

# MPI Support
from mpi4py import MPI

# 3D Modeling and Rendering
import vtk
import vtk.util.numpy_support as vtk_np

# Global Variables
processed_file = 'WD40.npy'
encoded_folder = 'WD40-SD512-ZZ'
dynamic_bounding = True
skeleton = True
curve_3d = 'zcurve_3D6.npy'
curve_2d = 'zcurve_2D9.npy'
range_ = [-100, 100]

# Verbose Settings
debug = True

################################################################################

def gen_mesh_voxels(pdb_data, bounds, sample_dim, debug=False):
    '''
    Method proceses PDB's atom data to create a space filling atomic model.
    The atomic model is then voxelized to generate a matrix for the space filling
    curve encoding.

    '''
    # Coordinate, Radius Information
    r = pdb_data[:,0].astype('float')
    x = pdb_data[:,1].astype('float')
    y = pdb_data[:,2].astype('float')
    z = pdb_data[:,3].astype('float')

    if debug:
        print("Generating Mesh...")
        start = time.time()

    # Generate Mesh For Protein
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(pdb_data)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x[i],y[i],z[i])
        sphere_source.SetRadius(r[i])
        sphere_source.Update()
        input1.ShallowCopy(sphere_source.GetOutput())
        append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    if debug:
        print time.time() - start, 'secs...'
        print("Voxelizing Mesh...")
        start = time.time()

    # Voxelize Mesh
    voxel_modeller = vtk.vtkVoxelModeller()
    voxel_modeller.SetInputConnection(clean_filter.GetOutputPort())
    voxel_modeller.SetSampleDimensions(sample_dim, sample_dim, sample_dim)
    x0, x1, y0, y1, z0, z1 = bounds
    voxel_modeller.SetModelBounds(x0, x1, y0, y1, z0, z1)
    voxel_modeller.SetMaximumDistance(0.01)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_output = voxel_modeller.GetOutput().GetPointData().GetScalars()
    voxel_array = vtk_np.vtk_to_numpy(voxel_output)
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    if debug:
        print time.time() - start, 'secs...'
        print("Filling Interiors...")
        start = time.time()

    # Fill Interiors
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    if debug: print time.time() - start, 'secs...'

    return filled_voxel_array

def gen_skeleton_voxels(pdb_data, max_, min_, res_):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.

    '''
    pdb_data = pdb_data[:,1:].astype('float')

    if debug:
        print("Generating Voxelizing Skeleton...")
        start = time.time()

    # Bin x, y, z Coordinates
    range_ = max_ - min_
    res_ = range_ / res_
    bins = [(i*res_) + min_ for i in range(int(range_/res_)+1)]
    x_binned = np.digitize(pdb_data[:, 0], bins) - 1
    y_binned = np.digitize(pdb_data[:, 1], bins) - 1
    z_binned = np.digitize(pdb_data[:, 2], bins) - 1
    indices = np.array([x_binned, y_binned, z_binned])
    indices = np.transpose(indices, (1, 0))

    # Get Unique Indices And Counts
    u_indices = {}
    for ind in indices:
        ind_ = tuple(ind.tolist())
        if ind_ in u_indices: u_indices[ind_] += 1
        else: u_indices[ind_] = 1

    # Generate 3D Array
    pdb_3d = np.zeros([int(range_/res_)+1 for i in range(3)])
    for ind in u_indices.keys(): pdb_3d[ind[0], ind[1], ind[2]] = 1

    if debug: print time.time() - start, 'secs...'

    return pdb_3d

def encode_3d_to_2d(array_3d, curve_3d, curve_2d, debug=False):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''
    if debug:
        print('Applying Space Filling Curves...')
        start = time.time()

    # Dimension Reduction Using Space Filling Curves to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

    if debug: print time.time() - start, 'secs...'

    return array_2d

def apply_rotation(pdb_data, rotation):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''
    if debug:
        print "Applying Rotation..."
        start = time.time()

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    if debug: print time.time() - start, 'secs...'

    return rotated_pdb_data

if __name__ == '__main__':

    # File Paths
    path_to_project = '../../'
    processed_file = path_to_project + 'data/inter/' + processed_file
    encoded_folder = path_to_project + 'data/final/' + encoded_folder + '/'
    curve_2d = path_to_project + 'data/source/SFC/'+ curve_2d
    curve_3d = path_to_project + 'data/source/SFC/'+ curve_3d

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if debug:
        print "MPI Info... Cores:", cores
        start = time.time()

    # MPI Cut Entries Per Node
    if debug: print "Loading Processed Entries..."
    entries = np.load(processed_file)
    entries = np.array_split(entries, cores)[rank]
    if debug:print "MPI Core", rank, ", Processing", len(entries), "Entries"

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)
    sample_dim = int(np.cbrt(len(curve_2d)))

    if debug:
        print "Init Time:", time.time() - start, "secs..."

    # Process Rotations
    for i in range(len(entries)):
        if debug: start = time.time()
        print "Processing", entries[i][0], "Rotation", entries[i][1]
        pdb_data = apply_rotation(entries[i][2], entries[i][3])
        if dynamic_bounding:
            dia = 0
            for channel in pdb_data:
                temp = np.amax(np.abs(channel[:, 1:])) + 2
                if temp > dia: dia = temp

        # Process Channels
        encoded_pdb_2d = []
        pdb_3d_model = []
        for j in range(len(pdb_data)):
            if debug: print('Processing Channel ' + str(j) + '...')
            pdb_data_res = pdb_data[j]
            if pdb_data_res is None: continue

            # Generate PDB Channel 3D Voxel Model
            if dynamic_bounding:
                bounds = [pow(-1,l+1)*dia for l in range(6)]
                if skeleton: pdb_3d_res = gen_skeleton_voxels(pdb_data_res, dia, -dia, sample_dim)
                else: pdb_3d_res = gen_mesh_voxels(pdb_data_res, bounds, sample_dim, debug=debug)
            else:
                bounds = range_ + range_ + range_
                if skeleton: pdb_3d_res = gen_skeleton_voxels(pdb_data_res, range_[0], range_[1], sample_dim)
                else: pdb_3d_res = gen_mesh_voxels(pdb_data_res, bounds, sample_dim, debug=debug)
            pdb_3d_model.append(pdb_3d_res)

            # Encode 3D Model with Space Filling Curve
            encoded_res_2d = encode_3d_to_2d(pdb_3d_res, curve_3d, curve_2d, debug=debug)
            encoded_pdb_2d.append(encoded_res_2d)
        encoded_pdb_2d = np.array(encoded_pdb_2d)
        encoded_pdb_2d = np.transpose(encoded_pdb_2d, (2,1,0))

        # Save Encoded PDB to Numpy Array File.
        if debug: print("Saving Encoded PDB...")
        file_path = encoded_folder + entries[i][0] + '-'+ str(entries[i][1]) +'.png'
        if not os.path.exists(encoded_folder): os.makedirs(encoded_folder)
        misc.imsave(file_path, encoded_pdb_2d)

        if debug:
            print "Encoding Time:", time.time() - start, "secs..."
            exit()
