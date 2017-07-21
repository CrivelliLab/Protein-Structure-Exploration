'''
ProcessPDBs.py
Updated: 7/20/17
[PASSING]

README:

The following script is used to generate array files with data points needed to
render and encode PBD files into 2D.

Global variables used to generate array file are defined under #- Global Variables.
'pdb_list' defines the .csv list of PDB chains which will be parsed. .csv must be
under data/start/PDB/.

Different channels of information can be accessed from the PDB by defining in
the 'sel_channels' variable. Note: these string identifiers must be valid Prody
tags.

Rotation matricies are generated for all permutations of rotations defined by 'theta'
degrees along the x, y, z axis.

Command Line Interface:

$ python ProcessPDBs.py [-h] pdb_list theta channels

The output array file will be saved under data/interim/ with the following naming
convention:

- <pdb_list>_t<theta>.npy

The array file will contain the following:

- (PDB_data_for_all_pdbs_in_folder, rotations)

PDB_data is structured as follows:

- (channel, radius, x, y, z)

'''
import os, argparse
from time import time
import numpy as np
import itertools as it

# PDB File Parsing
from prody import parsePDB, moveAtoms, confProDy
confProDy(verbosity='none')

#- Global Variables
pdb_list = ''
sel_channels = []
theta = 0

#- Verbose Settings
debug = False
pdb_list_usage = "PDB and chain ids list .csv file; .txt file must be in data/raw/PDB/"
channels_usage = "channels which will be encoded; comma seperated values"
theta_usage = "rotation angle in degrees"

################################################################################

# Hard Coded Knowledge
van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
'I' : 1.98, '' : 0} # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

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
    if debug: print "Parsing:", pdb_file
    molecule = parsePDB(pdb_file).select('protein')

    if chain: molecule = molecule.select('chain '+chain)

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

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_list', help=pdb_list_usage, type=str)
    parser.add_argument('theta', help=theta_usage, type=int)
    parser.add_argument('channels', help=channels_usage, type=str)
    args = vars(parser.parse_args())
    pdb_list = args['pdb_list']
    theta = args['theta']
    sel_channels = args['channels'].split(',')

    # File Paths
    pdb_folder = pdb_list.split('.')[0]
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    interim_file = '../../data/interim/' + pdb_folder + '_t' + str(theta)
    pdb_folder = '../../data/raw/PDB/' + pdb_folder + '/'

    # Read PDB File Names
    print "Read PDB Ids in:", pdb_folder[6:]
    if debug: t = time()
    pdb_files = []
    for line in sorted(os.listdir(pdb_folder)):
        if line.endswith('pdb.gz'): pdb_files.append(line)
    pdb_files = np.array(pdb_files)
    if debug: print time() - t, 'secs...'

    # Read File
    if debug: print("Reading PDB Chains...")
    with open('../../data/raw/PDB/'+pdb_list) as f:
        lines = f.readlines()
        pdb_chains = {}
        for x in lines:
            x = x.strip().split(',')
            if x[0].lower() in pdb_chains:
                pdb_chains[x[0].lower()] = pdb_chains[x[0].lower()] + x[1:]
            else: pdb_chains[x[0].lower()] = x[1:]

    # Generate Rotations
    if debug: print("Generating Rotations..."); t = time()
    base_rotations = []
    axis_list = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    theta_list = [(np.pi*(i*float(theta)/180))  for i in range(int(360/theta))]
    for axis in axis_list:
        base_rotations.append([])
        for theta in theta_list:
            rotation = get_rotation_matrix(axis, theta)
            base_rotations[-1].append(rotation)

    ## Generate Combinations of Base Rotations
    base_indices = [[i for i in range(len(base_rotations[j]))] for j in range(len(base_rotations))]
    indices = list(it.product(*base_indices))
    rotations = []
    for index in indices:
        comb_rotation = []
        for i in range(len(base_rotations)):
            comb_rotation.append(base_rotations[i][index[i]])
        rotations.append(comb_rotation)
    rotations = np.array(rotations)

    ## Dot Multiply Rotation Combinations
    combined_rotations = []
    for r in rotations: combined_rotations.append(r[2].dot(r[1].dot(r[0])))
    rotations = np.array(combined_rotations)
    if debug: print time() - t, 'secs...'

    # Parse PDB Data
    if debug: print("Processing PDBs..."); t = time()
    pdbs_data = []
    for pdb_file in pdb_files:
        chains = pdb_chains[pdb_file[:-7]]
        if len(chains) == 0:
            pdb_data = get_pdb_data(pdb_folder + pdb_file, None, channels=sel_channels, debug=False)
            pdbs_data.append([pdb_file.split('.')[0], pdb_data])
        else:
            for chain in chains:
                pdb_data = get_pdb_data(pdb_folder + pdb_file, chain, channels=sel_channels, debug=False)
                pdbs_data.append([pdb_file.split('.')[0]+chain, pdb_data])
    pdbs_data = np.array(pdbs_data)
    print len(pdbs_data)
    if debug: print time() - t, 'secs...'

    # Save Data
    if debug: print("Saving Data..."); t = time()
    data = np.array([pdbs_data, rotations])
    np.save(interim_file, data)
    if debug: print time() - t, 'secs...'
    print "Processed data saved in:", interim_file[6:] + '.npy'
