'''
Visualizations.py
Updated: 06/20/17

'''
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import vtk
from tvtk.api import tvtk
from tvtk.common import configure_input_data
from tqdm import tqdm

def display_3d_array(array_3d):
    '''
    Method displays 3d array.

    '''
    # Dislay 3D Mesh Rendering
    v = mlab.figure()
    for j in range(len(array_3d)):
        if j == 1: c = (1, 0, 0)
        elif j == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)

        # Coordinate Information
        xx, yy, zz = np.where(array_3d[j] >= 1)

        # Generate Mesh For Protein
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

        # Render Mesh
        pd = tvtk.to_tvtk(clean_filter.GetOutput())
        sphere_mapper = tvtk.PolyDataMapper()
        configure_input_data(sphere_mapper, pd)
        p = tvtk.Property(opacity=1.0, color=c)
        sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
        v.scene.add_actor(sphere_actor)

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
        r = pdb_data[j][:,0].astype('float')
        x = pdb_data[j][:,1].astype('float')
        y = pdb_data[j][:,2].astype('float')
        z = pdb_data[j][:,3].astype('float')

        # Generate Mesh For Protein
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(pdb_data[j])):
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

        # Render Mesh
        pd = tvtk.to_tvtk(clean_filter.GetOutput())
        sphere_mapper = tvtk.PolyDataMapper()
        configure_input_data(sphere_mapper, pd)
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
    pass
