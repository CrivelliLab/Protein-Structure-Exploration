'''
display_3d.py
Updated: 2/1/2018

Script is used to display 3D representation of data using VTK.

'''
from mayavi import mlab
import vtk
import os
import numpy as np
from tvtk.api import tvtk
from matplotlib.cm import *
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tvtk.common import configure_input_data

path = "../../data/KrasHras/Hras/1aa9_A"

################################################################################

def display_3d_array(array_3d, color_map, figure):
    '''
    Method displays 3d array.

    Param:
        array_3d - np.array
        attenmap - np.array

    '''
    cm = color_map

    # Dislay 3D Array Rendering
    v = figure
    for j in range(len(array_3d)):
        c = tuple(cm[j])

        # Coordinate Information
        xx, yy, zz = np.where(array_3d[j] > 0.0)
        xx *= 100
        yy *= 100
        zz *= 100

        # Generate Voxels For Protein
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(xx)):
            input1 = vtk.vtkPolyData()
            voxel_source = vtk.vtkCubeSource()
            voxel_source.SetCenter(xx[i],yy[i],zz[i])
            voxel_source.SetXLength(100)
            voxel_source.SetYLength(100)
            voxel_source.SetZLength(100)
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

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load Array
    array_3d = np.load(path + '/' + path.split('/')[-1] + '-3d.npz')['arr_0'].astype('int')

    # Draw Boundary Box
    curve = [[0,0,0], [0,64,0], [64,64,0], [64,0,0], [0,0,0], [0,0,64], [0,64,64],
            [64, 64, 64], [64,0,64], [0,0,64],[0,0,0],[0,64,0],[0,64,64],[0,64,0],
            [64,64,0], [64,64,64], [64,64,0], [64,0,0], [64,0,64]]
    curve= np.array(curve)
    curve *= 100 # Scaling in 3D Plot
    v = mlab.figure(bgcolor=(1.0,1.0,1.0))
    mlab.plot3d(curve[:,0], curve[:,1], curve[:,2], color=(0.5,0.5,1.0), tube_radius=15.0,
    opacity=0.5, figure=v)

    # Color Mapping
    n = array_3d.shape[-1]
    cm = [brg(float(i)/(n-1))[:3] for i in range(n)]

    # Display 3D array
    display_3d_array(np.transpose(array_3d,(3,0,1,2)),cm,v)
    mlab.show()
