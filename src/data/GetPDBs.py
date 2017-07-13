'''
GetPDBs.py
Updated: 07/12/17
[PASSING]

README:

The following script is used to fetch PDB files from the Protein Data Bank.

Global variables used to fetch the PDBs are defined under #- Global Variables.
'pdb_list_file' defines the .txt file containing list of PDB ids. .txt file
should be under data/raw/PDB/ and filename should be formated as follows:

- <label>_ids.txt

Command Line Interface:

$ python GetPDBs.py [-h] pdblist_ids.txt

A new folder will be created named <label> containing compressed PDB files under
data/raw/PDB/ .

'''
import os, argparse
from tqdm import tqdm

# PDB Fetching
from prody import fetchPDB, confProDy
confProDy(verbosity='none')

#- Global Variables
pdb_list_file = ''

#- Verbose Settings
debug = True
pdb_list_usage = "PDB ids list .txt file; .txt file must be in data/raw/PDB/"

################################################################################

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_list', help=pdb_list_usage, type=str,)
    args = vars(parser.parse_args())
    pdb_list_file = args['pdb_list']

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    pdbs_folder = '../../data/raw/PDB/'
    pdb_list = pdb_list_file.split('.')[0].split('_')[0]
    if not os.path.exists(pdbs_folder+pdb_list): os.makedirs(pdbs_folder+pdb_list)

    # Read File
    if debug: print("Reading PDB List...")
    with open(pdbs_folder+pdb_list_file) as f: pdb_ids = f.readlines()
    pdb_ids = [x.strip() for x in pdb_ids]
    if debug: print pdb_list, 'contains', len(pdb_ids), 'entries...'

    # Fetch PDBs
    if debug: print("Fetching PDBs..."); pbar = tqdm(total=len(pdb_ids))
    for i in range(len(pdb_ids)):
        fetchPDB(pdb_ids[i], compressed=True, folder=pdbs_folder+pdb_list)
        if debug: pbar.update(1)
    if debug: pbar.close(); print "PDBs saved in:", pdbs_folder+pdb_list
