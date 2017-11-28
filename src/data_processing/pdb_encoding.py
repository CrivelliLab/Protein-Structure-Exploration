'''
pdb_encoding.py
Updated:

'''
from PDB_DataGenerator import PDB_DataGenerator
from PDB_DataGenerator import *
import os
import numpy as np
from scipy.misc import imsave
import scipy
from time import time
from mpi4py import MPI
from tqdm import tqdm

data_folder = '../../data/raw/KRAS_HRAS/'
res_i = None
nb_rot = 500
chain = None

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        entries = []
        for class_fn in sorted(os.listdir(data_folder)):
            if os.path.isdir(data_folder+class_fn) and class_fn.lower().endswith('pdbs'):
                save_fn = class_fn.split('_')[0]
                if not os.path.exists(data_folder+save_fn): os.makedirs(data_folder+save_fn)
                for pdb_fn in sorted(os.listdir(data_folder+class_fn)):
                    for i in range(nb_rot):
                        if chain is None:
                            entries.append([data_folder+class_fn+'/'+pdb_fn, i, data_folder+save_fn+'/'+pdb_fn.split('.')[0]+'-r'+str(i)+'.png'])
                        else:
                            entries.append([data_folder+class_fn+'/'+pdb_fn, i, data_folder+save_fn+'/'+pdb_fn.split('.')[0]+'_'+chain+'-r'+str(i)+'.png'])
        entries = np.array(entries)
        np.random.seed(9999)
        np.random.shuffle(entries)
    else: entries = None
    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]

    # Intialize Data Generator
    pdb_datagen = PDB_DataGenerator(size=64, center=[0,0,0], resolution=1.0, thresh=0.95, nb_rots=nb_rot, map_to_2d=True,
                                    channels=[aliphatic_res, aromatic_res, neutral_res, acidic_res, basic_res, unique_res, alpha_carbons, beta_carbons])
    print(len(entries))

    for i in tqdm(range(len(entries))):
        # Entry Data
        pdb_path = entries[i][0]
        rot = int(entries[i][1])
        save_path = entries[i][2]
        c = save_path.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0]

        # Generate and Save Data
        pdb_array = pdb_datagen.generate_data(pdb_path, c, res_i, rot)
        if len(pdb_array) > 0: scipy.misc.toimage(pdb_array, cmin=0, cmax=255).save(save_path)

    print("Done")
