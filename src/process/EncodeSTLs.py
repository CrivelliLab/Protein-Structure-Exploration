'''
Encodeobjs.py
Updated: 6/23/17

README:

The following script is used to encode .OBJ files into 2D images using
spacefilling curves.

Global variables used to encode are defined under #- Global Variables. obj_folder
defines the folder containing .OBJ files. Folder must be under data/source.

encoded_folder defines the folder name for the encoded objects.Folder will be
created under data/final/.

dynamic_bounding defines whether dynamic or static bounding will be used to
discretize the object. If set to False, range_ defines the window along each axis
discretization will be done.

curve_3d and curve_2d define the spacefilling curves used for encoding for 3D to
1D and 1D to 2D respectively. Curves are under /data/source/SFC/.

The output image files are saved under data/final/<encoded_folder> as .png files
with the following naming convention:

<obj_id> - <rotation_index>.png

'''
import os, sys, time
import numpy as np
from scipy import misc, ndimage

# MPI
from mpi4py import MPI

# 3D Modeling and Rendering
import vtk
import vtk.util.numpy_support as vtk_np

# Global Variables
obj_folder = 'ShapeNetCore'
encoded_folder = 'ShapeNetCore-MD64'
dynamic_bounding = True
range_ = [-0.7, 0.7]
curve_3d = 'zcurve_3D4.npy'
curve_2d = 'fold_2D6.npy'

# Verbose Settings
debug = False

'''
# Defined Rotations
axis_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
theta_list = [(np.pi*i)/4  for i in range(8)]
'''

################################################################################

def gen_3d_obj(obj_file, bounds, sample_dim, debug=False):
    '''
    '''
    if debug: print("Generating Mesh...")

    # Generate Mesh For Protein
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()

    # Voxelize Mesh
    if debug: print("Voxelizing Mesh...")
    voxel_modeller = vtk.vtkVoxelModeller()
    voxel_modeller.SetInputConnection(reader.GetOutputPort())
    voxel_modeller.SetSampleDimensions(sample_dim, sample_dim, sample_dim)
    if bounds is None:
        bounds = reader.GetOutput().GetBounds()
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_rad = max([abs(x_range), abs(y_range), abs(z_range)])
        max_rad = max_rad/2
        temp = []
        temp.append(bounds[0]+(x_range/2)-max_rad)
        temp.append(bounds[1]-(x_range/2)+max_rad)
        temp.append(bounds[2]+(y_range/2)-max_rad)
        temp.append(bounds[3]-(y_range/2)+max_rad)
        temp.append(bounds[4]+(z_range/2)-max_rad)
        temp.append(bounds[5]-(z_range/2)+max_rad)
        bounds = temp
    x0, x1, y0, y1, z0, z1 = bounds
    voxel_modeller.SetModelBounds(x0, x1, y0, y1, z0, z1)
    voxel_modeller.SetMaximumDistance((x1- x0)/sample_dim)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_output = voxel_modeller.GetOutput().GetPointData().GetScalars()
    voxel_array = vtk_np.vtk_to_numpy(voxel_output)
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    '''
    xx, yy, zz = np.where(voxel_array >= 1)
    if len(xx) == 0: return voxel_array
    # Generate Mesh For Protein
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(xx)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkCubeSource()
        sphere_source.SetCenter(xx[i],yy[i],zz[i])
        sphere_source.SetXLength(1)
        sphere_source.SetYLength(1)
        sphere_source.SetZLength(1)
        sphere_source.Update()
        input1.ShallowCopy(sphere_source.GetOutput())
        append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.5, 0.5, 0) #  Background color dark red

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()
    '''
    # Fill Interiors
    if debug: print("Filling Interiors...")
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    return filled_voxel_array

def encode_3d_to_2d(array_3d, curve_3d, curve_2d, debug=False):
    '''
    Method proceses 3D voxel model and encodes into 2D image.

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

if __name__ == '__main__':

    # File Paths
    path_to_project = '../../'
    obj_folder = path_to_project + 'data/source/' + obj_folder + '/'
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

    # Read objs
    if debug: print "Loading Entries..."
    entries = []
    for line in sorted(os.listdir(obj_folder)):
        if line[0] != '.':
            for l in sorted(os.listdir(obj_folder+line)):
                if l[0] != '.': entries.append(line+'/'+l+'/model.obj')
    entries = np.array(entries)
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
        print('Processing Entry ' + str(i) + '...')

        # Process obj
        encoded_obj_2d = []
        obj_3d_model = []
        if debug: print('Processing obj...')
        if dynamic_bounding:
            obj_3d_res = gen_3d_obj(obj_folder+entries[i], None, sample_dim, debug=debug)
        else:
            bounds = range_ + range_ + range_
            obj_3d_res = gen_3d_obj(obj_folder+entries[i], bounds, sample_dim, debug=debug)
        obj_3d_model.append(obj_3d_res)
        obj_3d_model.append(obj_3d_res)
        obj_3d_model.append(obj_3d_res)

        # Encode 3D Model with Space Filling Curve
        encoded_res_2d = encode_3d_to_2d(obj_3d_res, curve_3d, curve_2d, debug=debug)
        encoded_obj_2d.append(encoded_res_2d)
        encoded_obj_2d.append(encoded_res_2d)
        encoded_obj_2d.append(encoded_res_2d)

        # Transpose Array
        encoded_obj_2d = np.array(encoded_obj_2d)
        encoded_obj_2d = np.transpose(encoded_obj_2d, (2,1,0))

        # Save Encoded obj to Numpy Array File.
        if debug: print("Saving Encoded obj...")
        file_path = encoded_folder + entries[i].split('/')[-3]+ '-'+ str(i) +'.png'
        if not os.path.exists(encoded_folder): os.makedirs(encoded_folder)
        misc.imsave(file_path, encoded_obj_2d)

        if debug: print "Processed in: ", time.time() - start, ' sec'

        if debug: exit()
