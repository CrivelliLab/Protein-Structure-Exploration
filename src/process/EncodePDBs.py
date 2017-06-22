'''
EncodePDB.py
Author: Rafael Zamora
Updated: 6/16/17

'''
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

# MPI
from mpi4py import MPI

# 3D Modeling and Rendering
import vtk
import vtk.util.numpy_support as vtk_np
#from tvtk.api import tvtk

# Global Variables
processed_file = '../data/Processed/WD40-20-21062017.npy'
encoded_folder = '../data/Encoded/WD40-SD64/'
dynamic_bounding = True
skeleton = True
curve_3d = '../data/SFC/zcurve_3D4.npy'
curve_2d = '../data/SFC/zcurve_2D6.npy'
range_ = [-100, 100]

# Verbose Settings
debug = False
visualize = False
profiling = False

def gen_mesh_voxels(pdb_data, bounds, sample_dim, debug=False):
    '''
    Method proceses PDB's atom data to create a matrix based 3d model.

    '''
    # Coordinate, Radius Information
    x = pdb_data[:,3].astype('float')
    y = pdb_data[:,2].astype('float')
    z = pdb_data[:,1].astype('float')
    s = pdb_data[:,0].astype('float')

    # Generate Mesh For Protein
    if debug: print("Generating Mesh...")
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(pdb_data)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x[i],y[i],z[i])
        sphere_source.SetRadius(s[i])
        sphere_source.Update()
        input1.ShallowCopy(sphere_source.GetOutput())
        if vtk.VTK_MAJOR_VERSION <= 5:
            append_filter.AddInputConnection(input1.GetProducerPort())
        else: append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    # Voxelize Mesh
    if debug: print("Voxelizing Mesh...")
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

    # Fill Interiors
    if debug: print("Filling Interiors...")
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    return voxel_array

def gen_skeleton_voxels(pdb_data, max_, min_, res_):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.
    '''
    pdb_data = pdb_data[:,1:].astype('float')

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

    return pdb_3d

def encode_3d_to_2d(array_3d, curve_3d, curve_2d, debug=False):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''
    if debug: print('Applying 3D to 1D Space Filling Curve...')

    # Dimension Reduction Using Space Filling Curve to 1D
    array_1d = np.zeros([len(curve_3d),])
    for i in range(len(curve_3d)):
        array_1d[i] = array_3d[curve_3d[i][0], curve_3d[i][1], curve_3d[i][2]]

    if debug: print('Applying 1D to 2D Space Filling Curve...')

    # Dimension Recasting Using Space Filling Curve to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(array_1d)):
        array_2d[curve_2d[i][0], curve_2d[i][1]] = array_1d[i]

    return array_2d

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

def display_3d_array(array_3d):
    '''
    Method displays 3d array.

    '''
    # Dislay 3D Voxel Rendering
    for i in range(len(array_3d)):
        if i == 1: c = (1, 0, 0)
        elif i == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)
        xx, yy, zz = np.where(array_3d[i] >= 1)
        mlab.points3d(xx, yy, zz, mode="cube", color=c)
    mlab.show()

def display_3d_mesh(pdb_data):
    '''
    '''
    # Dislay 3D Mesh Rendering
    v = mlab.figure()
    for j in range(len(pdb_data)):
        if j == 1: c = (1, 0, 0)
        elif j == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)

        # Coordinate, Radius Information
        x = pdb_data[j][:,3].astype('float')
        y = pdb_data[j][:,2].astype('float')
        z = pdb_data[j][:,1].astype('float')
        s = pdb_data[j][:,0].astype('float')

        # Generate Mesh For Protein
        for i in range(len(pdb_data[j])):
            sphere = tvtk.SphereSource(center=(x[i],y[i],z[i]), radius=s[i])
            sphere_mapper = tvtk.PolyDataMapper()
            configure_input_data(sphere_mapper, sphere.output)
            sphere.update()
            p = tvtk.Property(opacity=1.0, color=c)
            sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
            v.scene.add_actor(sphere_actor)

    mlab.show()

def display_3d_points(coords_3d):
    '''
    '''
    # Display 3D Mesh Rendering
    for i in range(len(coords_3d)):
        if i == 1: c = (1, 0, 0)
        elif i == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)
        # Coordinate, Radius Information
        x = coords_3d[i][:,3].astype('float')
        y = coords_3d[i][:,2].astype('float')
        z = coords_3d[i][:,1].astype('float')

        mlab.points3d(x, y, z, mode="sphere", color=c, scale_factor=0.5)
    mlab.show()

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    plt.figure()
    plt.imshow(array_2d, interpolation="nearest")
    plt.show()

if __name__ == '__main__':

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()
    if debug: print "MPI Info... Cores:", cores

    if profiling: start = time.time()

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

    if profiling: print time.time() - start, "secs..."

    # Process Rotations
    for i in range(len(entries)):
        if profiling: start = time.time()
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
        encoded_pdb_2d = misc.imresize(encoded_pdb_2d, (64,64,3))

        # Save Encoded PDB to Numpy Array File.
        if debug: print("Saving Encoded PDB...")
        file_path = encoded_folder + entries[i][0] + '-'+ str(entries[i][1]) +'.png'
        #np.savez_compressed(file_path, a=encoded_pdb_2d, b=rot)
        misc.imsave(file_path, encoded_pdb_2d)

        if profiling: print time.time() - start, "secs..."

        # Visualize PDB Data
        if visualize:
            print('Visualizing All Channels...')
            display_2d_array(encoded_pdb_2d)

        if debug: exit()
