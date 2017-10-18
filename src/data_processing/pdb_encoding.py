'''
pdb_encoding.py
Updated:

'''
from PDB_DataGenerator import PDB_DataGenerator
import os
import numpy as np
from itertools import product
from time import time
from scipy import misc

################################################################################

from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import brg
from tvtk.api import tvtk
from tvtk.common import configure_input_data
import vtk

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

def display_3d_array(array_3d):
    '''
    Method displays 3d array.

    Param:
        array_3d - np.array
        attenmap - np.array

    '''
    # Color Mapping
    n = len(array_3d)
    cm = [brg(float(i)/(n-1))[:3] for i in range(n)]

    # Dislay 3D Array Rendering
    v = mlab.figure(bgcolor=(1.0,1.0,1.0))
    for j in range(len(array_3d)):
        c = tuple(cm[j])

        # Coordinate Information
        xx, yy, zz = np.where(array_3d[j] > 0.0)

        # Generate Voxels For Protein
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(xx)):
            input1 = vtk.vtkPolyData()
            voxel_source = vtk.vtkCubeSource()
            voxel_source.SetCenter(xx[i],yy[i],zz[i])
            voxel_source.SetXLength(1)
            voxel_source.SetYLength(1)
            voxel_source.SetZLength(1)
            voxel_source.Update()
            input1.ShallowCopy(voxel_source.GetOutput())
            append_filter.AddInputData(input1)
        append_filter.Update()

        #  Remove Any Duplicate Points.
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputConnection(append_filter.GetOutputPort())
        clean_filter.Update()

        # Render Voxels
        pd = tvtk.to_tvtk(clean_filter.GetOutput())
        cube_mapper = tvtk.PolyDataMapper()
        configure_input_data(cube_mapper, pd)
        p = tvtk.Property(opacity=1.0, color=c)
        cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
        v.scene.add_actor(cube_actor)

    mlab.show()

if __name__ == '__main__':

    # Intialize Data Generator
    pdb_datagen = PDB_DataGenerator(size=64, center=[0,0,0], resolution=1.0)

    # Load Curves
    curve_3d = np.load('ModelNet/data/hilbert_3d_6.npy')
    curve_2d = np.load('ModelNet/data/hilbert_2d_9.npy')
    try: sample_dim = int(np.cbrt(len(curve_2d)))
    except: sample_dim = int(len(curve_2d) ** (1.0/3.0)) + 1

    # Parse PBD Atomic Data
    t = time()

    pdb_data = pdb_datagen.parse_pdb("/Users/rzamora/LBNL/Projects/Protein-Structure-Exploration/src/data_processing/1aa9_A.pdb")

    print "Parsing Time:", time() -t

    # Generate 3D Map
    t = time()

    pdb_data = pdb_datagen.remove_outlier_atoms(pdb_data)
    pdb_data, distances, indexes = pdb_datagen.calc_distances_from_voxels(pdb_data)
    indexes, values = pdb_datagen.apply_channels(pdb_data, distances, indexes)
    array_3d = pdb_datagen.generate_voxel_rep(indexes, values)

    print "Generation Time:",time() -t

    display_3d_array(array_3d)

    # Map Using SFC
    t = time()

    array_2d = pdb_datagen.map_3d_to_2d(array_3d, curve_3d, curve_2d)

    print time() - t

    array_2d = np.transpose(array_2d, (1,2,0))
    #plt.imshow(array_2d)
    #plt.show()
    display_2d_array(array_2d)
