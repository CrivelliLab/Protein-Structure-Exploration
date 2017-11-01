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

path = "../../data/raw/KRAS_HRAS/HRAS/1p2s_A-r0.png"

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

def map_2d_to_3d(array_2d, curve_3d, curve_2d):
    '''
    Method maps 2D PDB array into 3D array.
    '''
    s = int(np.cbrt(len(curve_3d)))
    array_3d = np.zeros([s,s,s, array_2d.shape[-1]])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        for j in range(array_2d.shape[-1]):
            array_3d[c3d[0], c3d[1], c3d[2], j] = array_2d[c2d[0], c2d[1], j]

    return array_3d

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    curve_2d = 'ModelNet/data/hilbert_2d_9.npy'
    curve_3d = 'ModelNet/data/hilbert_3d_6.npy'

    # Load Curves
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # Load Array
    array = imread(path)
    array = array[:,:,0] + (array[:,:,1] * 2**8) + (array[:,:,2] * 2**16)
    array = np.expand_dims(array.astype('>i8'), axis=-1)
    nb_chans = 8
    array = np.unpackbits(array.view('uint8'),axis=-1)[:,:,-nb_chans:]
    array = np.flip(array, axis=-1)

    #Map to 3D
    array_3d = map_2d_to_3d(array, curve_3d, curve_2d)
    display_3d_array(np.transpose(array_3d,(3,0,1,2)))
