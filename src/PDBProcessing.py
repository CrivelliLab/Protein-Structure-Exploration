'''
PDBProcessing.py
Authors: Rafael Zamora
Last Updated: 5/6/2017

This script is used to parse and process Protein Data Base entries.

'''
import os, wget
from prody import *
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
confProDy(verbosity='none')

from SpaceFillingCurves import display_3d_array

debug = True
residuals = [   'ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN', 'GLU', 'GLX',
                'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'UNK', 'VAL']

def get_pdb_data(pdb_id):
    '''
    Method fetches PDB from database and parses required residual information.

    '''
    # Fetch PDB And Parse FIle
    if debug: print("Fetching:", pdb_id)
    protein = parsePDB(fetchPDB(pdb_id, copy=False)).select('protein')
    moveAtoms(protein, to=np.zeros(3))
    #protein.select('resname ALA')
    coords = protein.getCoords()

    # Remove Local PDB File
    os.remove(pdb_id.lower() + '.pdb.gz')

    # Visualize Protein
    if debug: showProtein(protein)

    return coords

def process_pdb_data(pdb_data):
    '''
    Method processes protein residual structures into 3d array.

    '''
    # Bin x, y, z Coordinates
    res = 5
    max_ = 35
    min_ = -35
    range_ = max_ - min_
    bins = [(i*res) + min_ for i in range((range_//res)+1)]
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
        else: u_indices[ind_] = 0

    # Generate 3D Array With
    structure = np.zeros([(range_//res)+1 for i in range(3)])
    for ind in u_indices.keys(): structure[ind[0], ind[1], ind[2]] = u_indices[ind]
    if debug:
        structure = (structure / np.max(structure))
        display_3d_array(structure)
        display_3d_array(structure, mask=(0,2))

def vis_processed_pdb_data(processed_pdb_data):
    '''
    Method visualizes the processed PDB data.

    '''
    pass

if __name__ == '__main__':
    # Read Protein Data Bank IDs
    pdb_ids = []
    i = 0
    for line in open("pdb_ids.idx", 'r'):
        if i > 5: pdb_ids.append(line.split()[0])
        i += 1
    if debug: print("Total PDB Entries:", len(pdb_ids))

    # Process PDB Entries
    for pdb_id in pdb_ids:
        pdb_data = get_pdb_data(pdb_id)
        processed_pdb = process_pdb_data(pdb_data)
