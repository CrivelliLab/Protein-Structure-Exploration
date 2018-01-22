from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data
import vtk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import brg
import os
import numpy as np
from scipy.misc import imread

path = "data/1aa9_A"

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

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load Array
    array_3d = np.load(path + path.split('/')[-1] + '-3d.npz')['arr_0'].astype('int')
    print(array_3d.shape)

    # Display
    display_3d_array(np.transpose(array_3d,(3,0,1,2)))
