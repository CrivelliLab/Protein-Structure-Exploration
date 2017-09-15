'''
encode_pdb.py
Updated: 08/28/17

README:

'''
import os
import vtk
import numpy as np
from time import time
from mpi4py import MPI
import itertools as it
from shutil import copyfile
from scipy import misc, ndimage, stats
import vtk.util.numpy_support as vtk_np
from prody import fetchPDB, confProDy, parsePDB, moveAtoms
confProDy(verbosity='none')

#- Global Variables
pdb_list_folder = 'PSIBLAST/'
sel_channels = ['hydrophobic', 'polar', 'charged']
random_rotations = 'rot_512_9283764.npy'
curve_3d = 'hilbert_3d_6.npy'
curve_2d = 'hilbert_2d_9.npy'
bounds = [-32, 32]

# Hard Coded Knowledge
van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
'I' : 1.98, '' : 0} # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

################################################################################
debug = False

def get_pdb_data(pdb_file, chain, channels=[], debug=False):
    '''
    Method parses radii and coordinate information for each atom of different
    channel present in the PDB file, and returns as numpy array.

    For each atom in each channel the data is stored in the following order:
        -   (van_der_waal_radii, x, y, z)

    Note: Data is gathered for atoms belonging to only the protein structure.

    Param:
        pdb_file - str ; file path to pdb file
        channels - list(str) ; tags for channels selection
        debug - boolean ; display debug info

    Return:
        pdb_data - np.array ; multichanneled pdb atomic coordinates

    '''
    # Parse PDB File
    if debug: print("Parsing:", pdb_file)
    molecule = parsePDB(pdb_file).select('protein')
    molecule = molecule.select('chain '+chain)

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

    return np.array(pdb_data)

def get_rotation_matrix(axis, theta):
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Param:
        axis - list ; (x, y, z) axis coordinates
        theta - float ; angle of rotaion in radians

    Return:
        rotation_matrix - np.array

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

def apply_rotation(pdb_data, rotation, debug=False):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''
    if debug: print("Applying Rotation...")

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

    if debug: print("Generating Mesh...")

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

    if debug: print("Voxelizing Mesh...")

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

    if debug: print("Filling Interiors...")

    # Fill Interiors
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)
    filled_voxel_array = np.transpose(filled_voxel_array, (2,1,0))

    return filled_voxel_array

def map_3d_to_2d(array_3d, curve_3d, curve_2d, debug=False):
    '''
    Method maps 3D PDB array into 2D array.

    '''
    if debug: print('Applying Space Filling Curves...')

    # Dimension Reduction Using Space Filling Curves from 3D to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

    return array_2d

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    pdbs_folder = '../../data/interim/pdb/'
    pdb_list_folder = '../../data/raw/'+ pdb_list_folder
    curve_2d = '../../data/misc/'+ curve_2d
    curve_3d = '../../data/misc/'+ curve_3d
    random_rotations = '../../data/misc/'+ random_rotations

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)
    try: sample_dim = int(np.cbrt(len(curve_2d)))
    except: sample_dim = int(len(curve_2d) ** (1.0/3.0)) + 1

    # Load Rotations
    if debug: print("Generating Rotations...")
    rot_coord = np.load(random_rotations)
    rotations = []
    for c in rot_coord:
        angle = np.arctan(c[2]/np.sqrt((c[0]**2)+(c[1]**2)))
        axis = np.dot(np.array([[0,1],[-1,0]]), np.array([c[0],c[1]]))
        rot1 = get_rotation_matrix([axis[0],axis[1],0],angle)
        if c[0] < 0 and c[1] < 0 or c[0] < 0 and c[1] > 0 :
            rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]) + np.pi)
        else: rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]))
        rot = np.dot(rot1, rot2)
        rotations.append(rot)
    rotations = np.array(rotations)

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        if not os.path.exists(pdbs_folder): os.makedirs(pdbs_folder)

        # Parse PDB Data
        if debug: print("Reading PDB List...")
        pdbs_data = []
        for pdb_list in os.listdir(pdb_list_folder):
            if pdb_list.endswith('.csv'):
                if not os.path.exists(pdb_list_folder+pdb_list[:-4]):
                    os.makedirs(pdb_list_folder+pdb_list[:-4])
                with open(pdb_list_folder+pdb_list) as f:
                    lines = f.readlines()
                    for x in lines:
                        line = x.strip().split(',')
                        pdb_id = line[0].lower()
                        chains = line[1:]
                        if not os.path.isfile(pdbs_folder+pdb_id+'.pdb.gz'):
                            fetchPDB(pdb_id, compressed=True, folder=pdbs_folder)
                        for chain in chains:
                            #print(pdb_list[:-4], pdb_id, chain)
                            pdb_data = get_pdb_data(pdbs_folder + pdb_id+'.pdb.gz', chain, channels=sel_channels, debug=False)
                            pdbs_data.append([pdb_list[:-4]+'/'+pdb_id+chain, pdb_data])

        # Remove PDBs outside bounds
        if debug: print("Measuring PDB diameters...")
        bounded_pdbs_data = []
        for i in range(len(pdbs_data)):
            # Calculate Diameter
            dia = 0
            pdb_data  = pdbs_data[i][1]
            for channel in pdb_data:
                for atom in channel:
                    dist = np.linalg.norm(atom[1:])
                    if dist > dia: dia = dist
            dia *= 2
            if dia <= abs(bounds[1]-bounds[0])- 4:
                bounded_pdbs_data.append(pdbs_data[i])
        bounded_pdbs_data = np.array(bounded_pdbs_data)
        entries = []
        for i in range(len(bounded_pdbs_data)):
            for j in range(len(rotations)): entries.append([i,j])
        entries = np.array(entries)
        np.random.shuffle(entries)
    else:
        entries = None
        bounded_pdbs_data = None
    bounded_pdbs_data = comm.bcast(bounded_pdbs_data, root=0)
    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]

    # Process Rotations
    for pdb_i, r_i in entries:

        # Apply Rotation To PDB Data
        pdb_data = apply_rotation(bounded_pdbs_data[pdb_i][1], rotations[r_i])

        # Static Bounding
        b = bounds + bounds + bounds

        # Process Channels
        encoded_pdb_2d = []
        for j in range(len(pdb_data)):
            if debug: print('Processing Channel ' + str(j) + '...')
            pdb_channel = pdb_data[j]

            # Generate PDB Channel 3D Voxel Model
            pdb_channel_3d = gen_mesh_voxels(pdb_channel, b, sample_dim)

            # Encode 3D Model with Space Filling Curve
            encoded_channel_2d = map_3d_to_2d(pdb_channel_3d, curve_3d, curve_2d)
            encoded_pdb_2d.append(encoded_channel_2d)

        # Transpose 2D Mapping
        encoded_pdb_2d = np.array(encoded_pdb_2d)
        encoded_pdb_2d = np.transpose(encoded_pdb_2d, (2,1,0))

        # Save Encoded PDB to Numpy Array File.
        if debug: print("Saving Encoded PDB...")
        file_path = pdb_list_folder + str(bounded_pdbs_data[pdb_i][0]) + '-r'+ str(r_i) +'.png'
        misc.imsave(file_path, encoded_pdb_2d)
