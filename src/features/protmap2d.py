'''
protmap2d.py
Author: Rafael Zamora
Updated: 07/3/17

README:

Reason for defaults...

'''
import os, sys, time, argparse
import numpy as np
import itertools as it

# MPI Support
from mpi4py import MPI

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# 3D and 2D Modeling
import vtk
import vtk.util.numpy_support as vtk_np
from scipy import misc, ndimage

#- Hard Coded Knowledge
van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
                        'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
                        'I' : 1.98, '' : 0}
                        # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

################################################################################
""" Space Filling Curves

Implemented Curves:

- 3D Z-Order Curve
- 2D Z-Order Curve
- 3D Hilbert Curve
- 2D Hilbert Curve

"""

def zcurve_3d(order):
    '''
    Method returns 3d z-order curve of desired order.

    '''
    z_curve = []
    for i in range(pow(pow(2, order),3)):
        x = i
        x &= 0x09249249
        x = (x ^ (x >>  2)) & 0x030c30c3
        x = (x ^ (x >>  4)) & 0x0300f00f
        x = (x ^ (x >>  8)) & 0xff0000ff
        x = (x ^ (x >> 16)) & 0x000003ff

        y = i >> 1
        y &= 0x09249249
        y = (y ^ (y >>  2)) & 0x030c30c3
        y = (y ^ (y >>  4)) & 0x0300f00f
        y = (y ^ (y >>  8)) & 0xff0000ff
        y = (y ^ (y >> 16)) & 0x000003ff

        z = i >> 2
        z &= 0x09249249
        z = (z ^ (z >>  2)) & 0x030c30c3
        z = (z ^ (z >>  4)) & 0x0300f00f
        z = (z ^ (z >>  8)) & 0xff0000ff
        z = (z ^ (z >> 16)) & 0x000003ff

        z_curve.append([x, y, z])

    return np.array(z_curve)

def zcurve_2d(order):
    '''
    Method returns 2d z-order curve of deisred order.

    '''
    z_curve = []
    for i in range(pow(pow(2, order),2)):
        x = i
        x&= 0x55555555
        x = (x ^ (x >> 1)) & 0x33333333
        x = (x ^ (x >> 2)) & 0x0f0f0f0f
        x = (x ^ (x >> 4)) & 0x00ff00ff
        x = (x ^ (x >> 8)) & 0x0000ffff

        y = i >> 1
        y&= 0x55555555
        y = (y ^ (y >> 1)) & 0x33333333
        y = (y ^ (y >> 2)) & 0x0f0f0f0f
        y = (y ^ (y >> 4)) & 0x00ff00ff
        y = (y ^ (y >> 8)) & 0x0000ffff

        z_curve.append([x, y])

    return np.array(z_curve)

def hilbert_3d(order):
    '''
    Method returns 3d hilbert curve of desired order.

    '''
    def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
        if order == 0:
            xx = x + (xi + yi + zi)/3
            yy = y + (xj + yj + zj)/3
            zz = z + (xk + yk + zk)/3
            array.append((xx, yy, zz))
        else:
            gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

            gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
            yi/2, yj/2, yk/2, array)
            gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
            xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
            gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
            -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
            gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
            -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
            gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
            -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
            gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
            -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
            gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
            -zk/2, -xi/2, -xj/2, -xk/2, array)

    n = pow(2, order)
    hilbert_curve = []
    gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

    return np.array(hilbert_curve)

def hilbert_2d(order):
    '''
    Method returns 2d hilbert curve of desired order.

    '''
    def gen_2d(order, x, y, xi, xj, yi, yj, array):
        if order == 0:
            xx = x + (xi + yi)/2
            yy = y + (xj + yj)/2
            array.append((xx, yy))
        else:
            gen_2d(order-1, x, y, yi/2, yj/2, xi/2, xj/2, array)
            gen_2d(order-1, x + xi/2, y + xj/2, xi/2, xj/2, yi/2, yj/2, array)
            gen_2d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, array)
            gen_2d(order-1, x + xi/2 + yi, y + xj/2 + yj, -yi/2,-yj/2,-xi/2,-xj/2, array)

    n = pow(2, order)
    hilbert_curve = []
    gen_2d(order, 0, 0, n, 0, 0, n, hilbert_curve)

    return np.array(hilbert_curve)

################################################################################
""" PDB Parsing

Possible Channels:

- 'hydrophobic'
- 'polar'
- 'charged'

"""

def get_pdb_data(pdb_file, channels=[], verbose=False):
    '''
    Method parses radii and coordinate information for each atom of different
    channel present in the PDB file, and returns as numpy array.

    For each atom in each channel the data is stored in the following order:
        -   (van_der_waal_radii, x, y, z)

    Note: Data is gathered for atoms belonging to only the protein structure.

    '''
    # Parse PDB File
    if verbose:
        print "Parsing:", pdb_file
        start = time.time()
    molecule = parsePDB(pdb_file).select('protein')

    # Set Protein's Center Of Mass At Origin
    moveAtoms(molecule, to=np.zeros(3))

    # Gather Atom Information
    pdb_data = []
    for channel in channels:
        molecule_channel = molecule.select(channel)
        if molecule_channel is not None:
            channel_radii = [van_der_waal_radii[k] for k in molecule_channel.getElements()]
            channel_radii = np.expand_dims(channel_radii, 1)
            channel_coords = molecule_channel.getCoords()
            channel_data = np.concatenate([channel_radii, channel_coords], 1)
        else: channel_data = np.empty((1,4))
        pdb_data.append(channel_data)
    pdb_data = np.array(pdb_data)

    if verbose: print time.time() - start

    return pdb_data

def parse_PDBs(pdb_folder, channels):
    '''
    '''
    # Read PDB File Names
    pdb_files = []
    for line in sorted(os.listdir(pdb_folder)):
        if line.endswith('pdb.gz'): pdb_files.append(line)
    pdb_files = np.array(pdb_files)

    # Generate Parsed Data
    pdbs_data = []
    for i in range(len(pdb_files)):
        pdb_data = get_pdb_data(pdb_folder + pdb_files[i], channels=channels)
        pdbs_data.append(pdb_data)

    return np.array(pdbs_data), pdb_files

################################################################################
""" Data Augmentation

Implemented:

- Rotation permutations

"""

def get_rotation_matrix(axis, theta):
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    '''
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return rotation_matrix

def generate_rotations(axis_list, theta_list):
    '''
    Method generates all rotation combinations of base rotations defined by
    axis_list and theta_list.

    '''
    # Generate Base Rotations
    base_rotations = []
    for axis in axis_list:
        base_rotations.append([])
        for theta in theta_list:
            rotation = get_rotation_matrix(axis, theta)
            base_rotations[-1].append(rotation)

    # Generate Combinations of Base Rotations
    base_indices = [[i for i in range(len(base_rotations[j]))] for j in range(len(base_rotations))]
    indices = list(it.product(*base_indices))
    rotations = []
    for index in indices:
        comb_rotation = []
        for i in range(len(base_rotations)):
            comb_rotation.append(base_rotations[i][index[i]])
        rotations.append(comb_rotation)
    rotations = np.array(rotations)

    # Dot Multiply Rotation Combinations
    combined_rotations = []
    for r in rotations: combined_rotations.append(r[2].dot(r[1].dot(r[0])))

    return np.array(combined_rotations)

def apply_rotation(pdb_data, rotation, verbose=False):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''
    if verbose: print "Applying Rotation..."; start = time.time()

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    if verbose: print time.time() - start, 'secs...'

    return rotated_pdb_data

################################################################################
""" Modeling and Mapping

Implemented Models:

- Space Filling Atomic Model
- Skeletal Atomic Coordinate Model

"""

def gen_mesh_voxels(pdb_data, bounds, sample_dim, verbose=False):
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

    if verbose: print "Generating Mesh..."; start = time.time()

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

    if verbose:
        print time.time() - start, 'secs...'
        print "Voxelizing Mesh..."; start = time.time()

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

    if verbose:
        print time.time() - start, 'secs...'
        print "Filling Interiors..."; start = time.time()

    # Fill Interiors
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)
    filled_voxel_array = np.transpose(filled_voxel_array, (2,1,0))

    if verbose: print time.time() - start, 'secs...'

    return filled_voxel_array

def gen_skeleton_voxels(pdb_data, bounds, sample_dim, verbose=False):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.

    '''
    pdb_data = pdb_data[:,1:].astype('float')

    if verbose: print "Generating Voxelizing Skeleton..."; start = time.time()

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

    if verbose: print time.time() - start, 'secs...'

    return pdb_3d

def map_3d_to_2d(array_3d, curve_3d, curve_2d, verbose=False):
    '''
    Method maps 3D PDB array into 2D array.

    '''
    if verbose: print 'Applying Space Filling Curves...'; start = time.time()

    # Dimension Reduction Using Space Filling Curves from 3D to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

    if verbose: print time.time() - start, 'secs...'

    return array_2d

################################################################################
# Sbatch Generator

def gen_sbatch(args):
    '''
    '''
    pass

if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_folder', help="folder containing encodable PDBs", type=str)
    parser.add_argument('encoded_folder', help="target folder for encoded PDBs", type=str)
    parser.add_argument('-c3','--curve_3d', help="3d space filling curve used for encoding", type=str, default='hilbert_3d')
    parser.add_argument('-c2','--curve_2d', help="2d space filling curve used for encoding", type=str, default='hilbert_2d')
    parser.add_argument('-o3', '--order_3d', help="order of 3D curve", type=int, default=6)
    parser.add_argument('-o2', '--order_2d', help="order of 2D curve", type=int, default=9)
    parser.add_argument('-r', '--rotation', help="degree of rotations", type=int, default=45)
    parser.add_argument('-s', '--static', help="static bounds for encoding", type=list)
    parser.add_argument('-ch', '--channels', help="channels which will be encoded", type=list, default=['hydrophobic', 'polar', 'charged'])
    parser.add_argument('-sk', '--skeleton', help="skeletal model used for encoding", action="store_true")
    parser.add_argument('-gb', '--generate_sbatch', help="increase output verbosity", type=int)
    parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")

    # Set Variables
    args = vars(parser.parse_args())
    pdb_folder = args['pdb_folder']
    encoded_folder = args['encoded_folder']
    gen_3d = args['curve_3d']
    gen_2d = args['curve_2d']
    order_3d = args['order_3d']
    order_2d = args['order_2d']
    theta = args['rotation']
    static_bounds = args['static']
    channels = args['channels']
    generate_sbatch = args['generate_sbatch']
    skeleton = args['skeleton']
    verbose = args['verbose']

    if not pdb_folder.endswith('/'): pdb_folder += '/'
    if not encoded_folder.endswith('/'): encoded_folder += '/'

    np.random.seed(1984)

    # Create SBatch Script
    if generate_sbatch:
        if verbose: print "Generating Sbatch File..."
        close()

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        # Generate Curves - NP
        if verbose: print "Generating SPCs..."; start = time.time()

        curve_3d = globals()[gen_3d](order_3d)
        curve_2d = globals()[gen_2d](order_2d)

        if verbose: print time.time() - start, 'secs...'

        if verbose: print("Generating Rotations..."); start = time.time()

        # Generate Rotations - NP
        axis_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        theta_list = [(np.pi*(i*float(theta)/360))  for i in range(int(360/theta))]
        rotations = generate_rotations(axis_list, theta_list)

        if verbose: print time.time() - start, 'secs...'

        if verbose: print "Parsing PDBs..."; start = time.time()

        # Parse PDB Files - P
        pdbs_data, pdb_files = parse_PDBs(pdb_folder, channels)

        if verbose: print time.time() - start, 'secs...'

        # Entries
        entries = []
        for i in range(len(pdb_files)):
            for j in range(len(rotations)):
                entries.append([pdb_files[i].split('.')[0], i, j])
        entries = np.array(entries)
        np.random.shuffle(entries)
        entries = np.array_split(entries, cores)

    else:
        curve_3d = None
        curve_2d = None
        rotations = None
        pdbs_data = None
        entries = None

    # Broadcast Data From Root
    curve_3d = comm.bcast(curve_3d, root=0)
    curve_2d = comm.bcast(curve_2d, root=0)
    pdbs_data = comm.bcast(pdbs_data, root=0)
    rotations = comm.bcast(rotations, root=0)
    entries = comm.bcast(entries, root=0)[rank]
    sample_dim = int(np.cbrt(len(curve_2d)))

    # Encode PDB Data - P
    for i in range(len(entries)):
        if verbose:
            print "Processing", entries[i][0], "Rotation", entries[i][2]
            start = time.time()

        # Apply Rotation To PDB Data
        pdb_data = apply_rotation(pdbs_data[int(entries[i][1])], rotations[int(entries[i][2])])

        # Dynamic or Static Bounding
        if not static_bounds:
            dia = 0
            for channel in pdb_data:
                temp = np.amax(np.abs(channel[:, 1:])) + 2
                if temp > dia: dia = temp
            bounds = [pow(-1,l+1)*dia for l in range(6)]
        else: bounds = static_bounds + static_bounds + static_bounds

        # Encode Channels
        encoded_pdb_2d = []
        for j in range(len(pdb_data)):
            if verbose: print('Processing Channel ' + str(j) + '...')
            pdb_channel = pdb_data[j]
            if pdb_channel is None: continue

            # Generate PDB Channel 3D Voxel Model
            if skeleton: pdb_channel_3d = gen_skeleton_voxels(pdb_channel, bounds, sample_dim, verbose=verbose)
            else: pdb_channel_3d = gen_mesh_voxels(pdb_channel, bounds, sample_dim, verbose=verbose)

            # Encode 3D Model with Space Filling Curve
            encoded_channel_2d = map_3d_to_2d(pdb_channel_3d, curve_3d, curve_2d, verbose=verbose)
            encoded_pdb_2d.append(encoded_channel_2d)

        # Transpose 2D Mapping
        encoded_pdb_2d = np.array(encoded_pdb_2d)
        encoded_pdb_2d = np.transpose(encoded_pdb_2d, (2,1,0))
        encoded_pdb_2d = encoded_pdb_2d.astype('int')


        # Save Encoded PDB to Numpy Array File.
        if verbose: print("Saving Encoded PDB...")
        file_path = encoded_folder + entries[i][0] + '-'+ str(entries[i][2])
        np.savez_compressed(file_path, encoded_pdb_2d)

        if verbose: print "Encoding Time:", time.time() - start, "secs..."

        exit()
