'''
fetch_pdbs.py

'''

import os
from prody import pathPDBFolder, fetchPDB

data_folder = '../../data/raw/KRAS_HRAS/'

################################################################################

if __name__ == '__main__':

    # Set PATHs
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    pathPDBFolder('../../data/temp/')

    # Fetch PDBs
    for class_fn in sorted(os.listdir(data_folder)):
        if not class_fn.endswith('.csv'): continue

        # Make Folder For PDBs
        class_ = class_fn.split('.')[0] + '_PDBs'
        if not os.path.exists(data_folder+class_): os.mkdir(data_folder+class_)

        with open(data_folder+class_fn, 'r')as f:
            lines = f.readlines()
            for l in lines:
                # Get PDB IDs and Chain IDs
                l = l[:-1].split(',')
                pdb_id = l[0].lower()
                chains = ''.join(l[1:])
                pdb_fn = '_'.join([pdb_id, chains]) + '.pdb'

                # Fetch PDB File
                fetchPDB(pdb_id, compressed=False, folder=data_folder+class_)
                os.rename(data_folder+class_+'/'+pdb_id+'.pdb', data_folder+class_+'/'+pdb_fn)
