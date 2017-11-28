'''
fetch_pdbs.py

'''

import os
from prody import pathPDBFolder, fetchPDB
from mpi4py import MPI
import numpy as np

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

    if rank == 0:
        entries = []
        for class_fn in sorted(os.listdir(data_folder)):
            if not class_fn.endswith('.csv'): continue

            # Make Folder For PDBs
            class_ = class_fn.split('.')[0] + '_pdbs'
            if not os.path.exists(data_folder+class_): os.mkdir(data_folder+class_)

            with open(data_folder+class_fn, 'r')as f:
                lines = f.readlines()
                for l in lines:
                    # Get PDB IDs and Chain IDs
                    l = l[:-1].split(',')
                    pdb_id = l[0].lower()
                    chain = l[1]
                    entries.append([data_folder+class_, pdb_id, chain])

        np.random.seed(9999)
        np.random.shuffle(entries)
    else: entries = None
    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]

    # Fetch PDBs
    for e in entries:
        fetchPDB(e[1], compressed=False, folder=e[0])
        os.rename(e[0]+'/'+e[1]+'.pdb', e[0]+'/'+e[1]+'_'+e[2]+'.pdb')
