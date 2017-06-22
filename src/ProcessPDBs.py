'''
ProcessPDBs.py
Updated: 06/21/17
Log:
- Bug in apply_rotation(): deep copy fixed issue
'''
import os, sys
import numpy as np
import itertools as it

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# Global Variables
seed = 21062017
sample = 20
pdb_folder = '../data/PDB/WD40/'
processed_file = '../data/Processed/' + pdb_folder.split('/')[-2] +'-'+str(sample)+'-'+str(seed)
sel_channels = ['hydrophobic', 'polar', 'charged']

# Verbose Settings
debug = True

# Defined Rotations
axis_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
theta_list = [(np.pi*i)/4  for i in range(8)]

# Hard Coded Knowledge
residuals = [   'ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN', 'GLU', 'GLX',
                'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'UNK', 'VAL']
elem_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
                'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
                'I' : 1.98, '' : 0}

def get_pdb_data(pdb_file, channels=[], debug=False):
    '''
    Method parses radii and coordinate information for each atom of different
    channel present in the PDB file, and returns as numpy array.
    '''
    # Parse PDB File
    if debug: print "Parsing:", pdb_file
    molecule = parsePDB(pdb_file).select('protein')

    # Set Protein's Center Of Mass At Origin
    moveAtoms(molecule, to=np.zeros(3))

    # Gather Atom Information
    pdb_data = []
    for channel in channels:
        channel_ = molecule.select(channel)
        if channel_ is not None:
            channel_radii = [elem_radii[k] for k in channel_.getElements()]
            channel_radii = np.expand_dims(channel_radii, 1)
            channel_coords = channel_.getCoords()
            channel_data = np.concatenate([channel_radii, channel_coords], 1)
        else: channel_data = None
        pdb_data.append(channel_data)
    pdb_data = np.array(pdb_data)

    return pdb_data

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

if __name__ == '__main__':

    np.random.seed(seed)

    # Read PDB File Names
    pdb_files = []
    for line in sorted(os.listdir(pdb_folder)): pdb_files.append(line)
    if debug: print "Processing PDBs in:", pdb_folder

    pdb_files = np.array(pdb_files)
    np.random.shuffle(pdb_files)
    pdb_files = pdb_files[:sample]

    # Generate Rotations
    if debug: print("Generating Rotations...")
    base_rotations = []
    for axis in axis_list:
        base_rotations.append([])
        for theta in theta_list:
            rotation = get_rotation_matrix(axis, theta)
            base_rotations[-1].append(rotation)

    base_indices = [[i for i in range(len(base_rotations[j]))] for j in range(len(base_rotations))]
    indices = list(it.product(*base_indices))
    rotations = []
    for index in indices:
        comb_rotation = []
        for i in range(len(base_rotations)):
            comb_rotation.append(base_rotations[i][index[i]])
        rotations.append(comb_rotation)
    rotations = np.array(rotations)

    temp = []
    for rot in rotations:
        temp.append(rot[2].dot(rot[1].dot(rot[0])))
    rotations = np.array(temp)

    # Generate Processed Data
    if debug: print("Processing PDBs...")
    processed_data = []
    for pdb_file in pdb_files:
        pdb_data = get_pdb_data(pdb_folder + pdb_file, channels=sel_channels, debug=debug)
        if debug: print("Applying Rotations...")
        for i in range(len(rotations)):
            processed_data.append([pdb_file.split('.')[0], i, pdb_data, rotations[i]])
    processed_data = np.array(processed_data)

    # Shuffle Data And Save
    np.random.shuffle(processed_data)
    np.save(processed_file, processed_data)
