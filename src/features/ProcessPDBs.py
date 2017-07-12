'''
ProcessPDBs.py
Updated: 06/26/17

README:

The following script is used to generate array files with data points needed to
render and encode PBD files into 2D images.

Global variables used to generate array file are defined under #- Global Variables.
pdb_folder defines the location of the PDBs which will be parsed. Folder must be
under data/start/PDB/.

Different channels of information can be accessed from the PDB by defining in
the sel_channels variable. Note: these string identifiers must be valid Prody
tags.

Rotation matricies are generated for all permutations of rotations defined by theta.
Currently set to generate all permuations of 45 degree turns along the x, y, z axis.

Any hardcoded data points not provided directly from the PDBs are defined under
#- Hard Coded Knowledge. Dictionary of Van Der Waal radii can be found here.

The output array file will be saved under data/interim/ with a file name corresponding
to the pdb_folder used to generate the file. Each entry in the array file will
contain the following:

(pdbs_data, rotations)

PDB_data is structured as follows:

(channel, radius, x, y, z)

'''
import os, argparse
from time import time
import numpy as np
import itertools as it

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

#- Global Variables
pdb_folder = 'P01111P01112P01116neg'
sel_channels = ['hydrophobic', 'polar', 'charged']
theta = 45

#- Verbose Settings
debug = True

################################################################################

# Hard Coded Knowledge
van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
'I' : 1.98, '' : 0} # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

def get_pdb_data(pdb_file, channels=[], debug=False):
    '''
    Method parses radii and coordinate information for each atom of different
    channel present in the PDB file, and returns as numpy array.

    For each atom in each channel the data is stored in the following order:
        -   (van_der_waal_radii, x, y, z)

    Note: Data is gathered for atoms belonging to only the protein structure.

    '''
    # Parse PDB File
    if debug: print "Parsing:", pdb_file
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

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-pf', '--pdb_folder',
                        help="Folder Containing PDBs", type=str, default=None)
    parser.add_argument('-th', '--theta',
                        help="Rotation Angle in Degrees", type=int, default=None)
    parser.add_argument('-ch', '--channels',
                        help="channels which will be encoded; comma seperated values",
                        type=str, default=None)
    args = vars(parser.parse_args())
    if args['pdb_folder']: pdb_folder = args['pdb_folder']
    if args['theta']: theta = args['theta']
    if args['channels']: sel_channels = args['channels'].split(',')

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    interim_file = '../../data/interim/' + pdb_folder + '_t' + str(theta)
    pdb_folder = '../../data/raw/PDB/' + pdb_folder + '/'

    # Read PDB File Names
    if debug: print "Read PDB Ids in:", pdb_folder; t = time()
    pdb_files = []
    for line in sorted(os.listdir(pdb_folder)):
        if line.endswith('pdb.gz'): pdb_files.append(line)
    pdb_files = np.array(pdb_files)
    if debug: print time() - t, 'secs...'

    # Generate Base Rotations
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

    # Generate Processed Data
    if debug: print("Processing PDBs..."); t = time()
    pdbs_data = []
    for pdb_file in pdb_files:
        pdb_data = get_pdb_data(pdb_folder + pdb_file, channels=sel_channels, debug=False)
        pdbs_data.append([pdb_file.split('.')[0], pdb_data])
    pdbs_data = np.array(pdbs_data)
    if debug: print time() - t, 'secs...'

    # Save Data
    if debug: print("Saving Data..."); t = time()
    data = np.array([pdbs_data, rotations])
    np.save(interim_file, data)
    if debug: print time() - t, 'secs...'
