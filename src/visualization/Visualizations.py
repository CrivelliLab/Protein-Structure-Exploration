'''
Visualizations.py
Author: Rafael Zamora
Updated: 07/10/17

README:

The following script is used to render visualizations of various 2D and 3D
models.

Global variables used to run renderings are defined under #- Global Variables.

'''
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import jet
from tvtk.api import tvtk
from tvtk.common import configure_input_data
from tqdm import tqdm
import numpy as np
import vtk
from scipy import misc

#- Global Variables
pdb_id = '1aa9'
rot_id = 0
curve_3d = 'hilbert_3D6.npy'
curve_2d = 'hilbert_2D9.npy'
encoded_folder = 'RAS-MD512-HH'
processed_file = 'RAS_t45.npy'

render_attenmap = False

#- Verbose Settings
debug = True

################################################################################

def display_2d_array(array_2d, attenmap=None):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    n = array_2d.shape[-1]
    cm = [jet(float(i)/n)[:3] for i in range(n)]
    for i in range(n):
        if i == 0: cmap = ListedColormap([[0,0,0,0.5], cm[i][:3]])
        else: cmap = ListedColormap([[0,0,0,0], cm[i][:3]])
        plt.imshow(array_2d[:,:,i], cmap=cmap, interpolation='nearest')
    plt.show()

def display_3d_array(array_3d, attenmap=None):
    '''
    Method displays 3d array.

    '''
    # Color Mapping
    n = len(array_3d)
    cm = [jet(float(i)/n)[:3] for i in range(n)]

    # Dislay 3D Array Rendering
    v = mlab.figure()
    for j in range(len(array_3d)):
        c = cm[j]

        # Coordinate Information
        xx, yy, zz = np.where(array_3d[j] >= 1)

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

def display_3d_model(pdb_data, skeletal=False, attenmap=None):
    '''
    Method renders space-filling atomic model of PDB data.

    '''

    # Color Mapping
    n = len(pdb_data)
    cm = [jet(float(i)/n)[:3] for i in range(n)]

    # Dislay 3D Mesh Rendering
    v = mlab.figure()
    for j in range(len(pdb_data)):
        c = cm[j]

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
            if skeletal: sphere_source.SetRadius(0.1)
            else: sphere_source.SetRadius(r[i])
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

def map_2d_to_3d(array_2d, curve_3d, curve_2d):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''

    # Dimension Reduction Using Space Filling Curves to 2D
    s = int(np.cbrt(len(curve_3d)))
    array_3d = np.zeros([s,s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[1], c2d[0]]

    return array_3d

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

if __name__ == '__main__':
    # File Paths
    path_to_project = '../../'
    curve_2d = path_to_project + 'data/raw/SFC/'+ curve_2d
    curve_3d = path_to_project + 'data/raw/SFC/'+ curve_3d
    encoded_folder = path_to_project + 'data/processed/PDB/' + encoded_folder + '/'
    processed_file = path_to_project + 'data/interim/' + processed_file
    pdb = pdb_id + '-' + str(rot_id) + '.png'

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # Load Encoded PDB
    if debug: print("Loading Encoded PDB...")
    encoded_pdb = misc.imread(encoded_folder + pdb)
    encoded_pdb = encoded_pdb.astype('float')/255.0
    decoded_pdb = []
    for i in range(encoded_pdb.shape[-1]):
        channel = encoded_pdb[:,:,i]
        decoded_channel = map_2d_to_3d(channel, curve_3d, curve_2d)
        decoded_pdb.append(decoded_channel)
    decoded_pdb = np.array(decoded_pdb)

    # Load PDB Data
    if debug: print("Loading PDB Data...")
    data = np.load(processed_file)
    rot = data[1][rot_id]
    pdb_data = None
    for d in data[0]:
        if d[0] == pdb_id: pdb_data = d[1]
    pdb_data = apply_rotation(pdb_data, rot)

    # Load Attention Map
    attenmap_2d = None
    attenmap_3d = None
    if render_attenmap:
        if debug: print("Loading Attention Map...")
        attenmap_2d = misc.imread('../../data/valid/attenmaps/' + pdb)
        attenmap_2d = img.astype('float')/255.0
        #img[img < 0.2] = 0
        attenmap_3d = map_2d_to_3d(attenmap_2d, curve_3d, curve_2d)

        # Calculate Diameter
        dia = 0
        for channel in pdb_data:
            temp = np.amax(np.abs(channel[:, 1:])) + 2
            if temp > dia: dia = temp

    # Render Visualizations
    if debug: print("Rendering Models...")
    display_3d_model(pdb_data, skeletal=True, attenmap=attenmap_3d)
    display_3d_model(pdb_data, attenmap=attenmap_3d)
    display_3d_array(decoded_pdb, attenmap=attenmap_3d)
    display_2d_array(encoded_pdb, attenmap=attenmap_2d)
