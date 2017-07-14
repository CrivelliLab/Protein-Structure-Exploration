'''
EncodePDBs.py
Updated: 7/12/17
[NOT PASSING] - Unimplemented Functionality
                |- Profiling Tool

README:

The following script is used to encode PDB data points into 2D using spacefilling
curves.

Global variables used to encode are defined under #- Global Variables.
'processed_file' defines the array file containing processed pdb data and rotation permutations.
Array file must be under data/interim/.

'dynamic_bounding' defines whether dynamic or static bounding will be used to
discretize the data. If set to False, 'bounds' defines the window along each axis
discretization will be done.

'skeleton' defines whether the encoding will be done without using radial data.
If set to False, a space filling atomic model will be rendered to be used in
discretization.

'curve_3d' and 'curve_2d' define the spacefilling curves used for encoding for 3D to
1D and 1D to 2D respectively. Curves are under /data/raw/SFC/.

Command Line Interface:

$ python EncodePDBs.py [-h] [-sk] [-sb STATIC_BOUNDS] processed_file curve_3d curve_2d

The output 2D files are saved under data/processed/<encoded_folder> where
<encoded_folder> follows the naming convention

- <processed_file>-(S or M for 'skeleton')()

2D files are saved in the folder with the following naming convention:

<pdb_id> - <rotation_index>.png

'''
import os, argparse
from time import time
import numpy as np
from scipy import misc, ndimage

# MPI Support
from mpi4py import MPI

# 3D Modeling and Rendering
import vtk
import vtk.util.numpy_support as vtk_np

#- Global Variables
processed_file = ''
skeleton = False
dynamic_bounding = True
bounds = [-70, 70]
curve_3d = 'hilbert_3d6.npy'
curve_2d = 'hilbert_2d9.npy'

#- debug Settings
debug = False
processed_file_usage = "processed PDB .npy file"
skeleton_usage = "use skeletal model for encoding"
static_bounds_usage = "static bounds for encoding; comma seperated values"
curve_3d_usage = "3d SFC used for encoding"
curve_2d_usage = "2d SFC used for encoding"

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

    if debug: print "Generating Mesh..."; t = time()

    # Generate Mesh For Protein
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(pdb_data)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x[i],y[i],z[i])
        sphere_source.SetRadius(r[i])
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

    if debug:
        print time() - t, 'secs...'
        print "Voxelizing Mesh..."; t = time()

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
        print time() - t, 'secs...'
        print "Filling Interiors..."; t = time()

    # Fill Interiors
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)
    filled_voxel_array = np.transpose(filled_voxel_array, (2,1,0))

    if debug: print time() - t, 'secs...'

    return filled_voxel_array

def gen_skeleton_voxels(pdb_data, bounds, sample_dim, debug=False):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.

    '''
    pdb_data = pdb_data[:,1:].astype('float')

    if debug: print "Generating Voxelizing Skeleton..."; t = time()

    # Bin x, y, z Coordinates
    min_ = bounds[0]
    max_ = bounds[1]
    range_ = max_ - min_
    res_ = range_ / sample_dim
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

    if debug: print time() - t, 'secs...'

    return pdb_3d

def map_3d_to_2d(array_3d, curve_3d, curve_2d, debug=False):
    '''
    Method maps 3D PDB array into 2D array.

    '''
    if debug: print 'Applying Space Filling Curves...'; t = time()

    # Dimension Reduction Using Space Filling Curves from 3D to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

    if debug: print time() - t, 'secs...'

    return array_2d

def apply_rotation(pdb_data, rotation, debug=False):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''
    if debug: print "Applying Rotation..."; t = time()

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    if debug: print time() - t, 'secs...'

    return rotated_pdb_data

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('processed_file', help=processed_file_usage, type=str)
    parser.add_argument('curve_3d', help=curve_3d_usage, type=str)
    parser.add_argument('curve_2d', help=curve_2d_usage, type=str)
    parser.add_argument('-sk', '--skeletal', help=skeleton_usage, action="store_true")
    parser.add_argument('-sb', '--static_bounds', help=static_bounds_usage, type=str, default=None)
    args = vars(parser.parse_args())
    processed_file = args['processed_file']
    curve_3d = args['curve_3d']
    curve_2d = args['curve_2d']
    if args['skeletal']: skeleton = True
    if args['static_bounds']:
        dynamic_bounding = False
        bounds = [int(i) for i in args['static_bounds'].split(',')]

    # Encoded Folder Name
    encoded_folder = processed_file.replace('_','-')[:-4] + '-'
    if skeleton: encoded_folder += 'S'
    else: encoded_folder += 'M'
    if dynamic_bounding: encoded_folder += 'D-'
    else: encoded_folder += 'S-'
    encoded_folder += curve_3d[0]
    encoded_folder += curve_2d[0]
    encoded_folder = encoded_folder.upper()

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    processed_file = '../../data/interim/' + processed_file
    encoded_folder = '../../data/processed/tars/' + encoded_folder
    curve_2d = '../../data/raw/SFC/'+ curve_2d
    curve_3d = '../../data/raw/SFC/'+ curve_3d

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        print "Encoding:", processed_file[6:]
        print "MPI Cores:", cores;
    if debug: t = time()

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)
    try: sample_dim = int(np.cbrt(len(curve_2d)))
    except: sample_dim = int(len(curve_2d) ** (1.0/3.0)) + 1
    encoded_folder += str(int(np.sqrt(len(curve_2d)))) + '/'

    # Load Data
    if debug: print("Loading PDB Data and Rotations...")
    data = np.load(processed_file)
    pdbs_data = data[0]
    rotations = data[1]

    # MPI Cut Entries Per Node
    if debug: print "Distributing Entries..."
    entries = []
    for i in range(len(pdbs_data)):
        for j in range(len(rotations)): entries.append([i,j])
    entries = np.array(entries)
    np.random.shuffle(entries)
    entries = np.array_split(entries, cores)[rank]
    if debug:
        print "MPI Core", rank, "Encoding", len(entries), "Entries..."
        print "Init Time:", time() - t, "secs..."

    # Process Rotations
    for pdb_i, r_i in entries:
        if debug: print "Processing", pdbs_data[pdb_i][0], "Rotation", r_i; t = time()

        # Apply Rotation To PDB Data
        pdb_data = apply_rotation(pdbs_data[pdb_i][1], rotations[r_i])

        # Dynamic Or Static Bounding
        if dynamic_bounding:
            dia = 0
            for channel in pdb_data:
                temp = np.amax(np.abs(channel[:, 1:])) + 2
                if temp > dia: dia = temp
            bounds = [pow(-1,l+1)*dia for l in range(6)]
        else: bounds = bounds + bounds + bounds

        # Process Channels
        encoded_pdb_2d = []
        for j in range(len(pdb_data)):
            if debug: print('Processing Channel ' + str(j) + '...')
            pdb_channel = pdb_data[j]
            if pdb_channel is None: continue

            # Generate PDB Channel 3D Voxel Model
            if skeleton: pdb_channel_3d = gen_skeleton_voxels(pdb_channel, bounds, sample_dim, debug=debug)
            else: pdb_channel_3d = gen_mesh_voxels(pdb_channel, bounds, sample_dim, debug=debug)

            # Encode 3D Model with Space Filling Curve
            encoded_channel_2d = map_3d_to_2d(pdb_channel_3d, curve_3d, curve_2d, debug=debug)
            encoded_pdb_2d.append(encoded_channel_2d)

        # Transpose 2D Mapping
        encoded_pdb_2d = np.array(encoded_pdb_2d)
        encoded_pdb_2d = np.transpose(encoded_pdb_2d, (2,1,0))

        # Save Encoded PDB to Numpy Array File.
        print("Saving Encoded PDB...")
        file_path = encoded_folder + str(pdbs_data[pdb_i][0]) + '-r'+ str(r_i) +'.png'
        if not os.path.exists(encoded_folder): os.makedirs(encoded_folder)
        misc.imsave(file_path, encoded_pdb_2d)

        if debug: print "Encoding Time:", time() - t, "secs..."; exit()

    if rank == 0: print "Encoding saved in:", encoded_folder[6:]
