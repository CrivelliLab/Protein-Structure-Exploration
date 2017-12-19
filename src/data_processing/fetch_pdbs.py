'''
fetch_pdbs.py
Updated: 11/28/17

Script fetches pdbs from Protein Data Bank as defined in class.csv files in data_folder
path. class.csv files includes pdb identifier and chain identifier pairs. PDBs are
save into folders corresponding to the .csv files.

'''

import os
from prody import pathPDBFolder, fetchPDB
from mpi4py import MPI
import numpy as np

# Data Folder Path
data_folder = '../../data/raw/KRAS_HRAS/'

###############################################################################

if __name__ == '__main__':

    # Set PATHs
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    pathPDBFolder('../../data/temp/')

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    # MPI Task Distribution
    if rank == 0:
        tasks = []

        # Iterate over PDB class .csv files
        for class_fn in sorted(os.listdir(data_folder)):
            if not class_fn.endswith('.csv'): continue

            # Make folder for PDB class
            class_ = class_fn.split('.')[0] + '_pdbs'
            if not os.path.exists(data_folder+class_): os.mkdir(data_folder+class_)

            # Iterate over PDB id and chain id pairs
            with open(data_folder+class_fn, 'r')as f:
                lines = f.readlines()
                for l in lines:

                    # Parse PDB IDs and chain IDs
                    l = l[:-1].split(',')
                    pdb_id = l[0].lower()
                    chain_id = l[1]
                    tasks.append([data_folder+class_, pdb_id, chain_id])

        # Shuffle for Random Distribution
        np.random.seed(9999)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Fetch PDBs
    for t in tasks:

        # Task IDs
        folder_ = t[0]
        pdb_id = t[1]
        chain_id = t[2]

        # Fetch PDB file and rename with task IDs
        fetchPDB(pdb_id, compressed=False, folder=folder_)
        os.rename(folder_+'/'+pdb_id+'.pdb', folder_+'/'+pdb_id+'_'+chain_id+'.pdb')
