'''
GetPDBs.py
Author: Rafael Zamora
Updated: 07/08/17

README:

The following script is used to fetch PDB files from the Protein Data Bank.

Global variables used to fetch the PDBs are defined under #- Global Variables.
'pdb_list_file' defines the .txt file containing list of PDB ids. .txt file should
be under data/raw/PDB/ and filename should be formated as follows:

<label>_ids.txt

A new folder will be created named <label> containing compressed PDB files under
data/raw/PDB/ .

'''
import os
from prody import fetchPDB
from tqdm import tqdm

#- Global Variables
pdb_list_file = 'P01111-P01112-P01116-pos_ids.txt'

#- Verbose Settings
debug = True

################################################################################

if __name__ == '__main__':

    # File Paths
    pdbs_folder = '../../data/raw/PDB/'
    pdb_list = pdb_list_file.split('_')[0]
    if not os.path.exists(pdbs_folder+pdb_list): os.makedirs(pdbs_folder+pdb_list)

    # Read File
    if debug: print("Reading PDB List...")
    with open(pdbs_folder+pdb_list_file) as f: pdb_ids = f.readlines()
    pdb_ids = [x.strip() for x in pdb_ids]
    if debug: print pdb_list, 'contains', len(pdb_ids), 'entries...'

    # Fetch PDBs
    if debug: print("Fetching PDBs..."); pbar = tqdm(total=len(pdb_ids))
    for i in tqdm(range(len(pdb_ids))):
        fetchPDB(pdb_ids[i], compressed=True, folder=pdbs_folder+pdb_list)
        if debug: pbar.update(1)
    if debug: pbar.close()
