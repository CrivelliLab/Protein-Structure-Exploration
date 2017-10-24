'''
pdb_encoding.py
Updated:

'''
from PDB_DataGenerator import PDB_DataGenerator
from PDB_DataGenerator import hydrophobic_res, polar_res, charged_res, alpha_carbons, beta_carbons
import os
import numpy as np
from scipy.misc import imsave, imread, imresize
import scipy
from time import time
import matplotlib.pyplot as plt
from binvox_io import write_binvox, read_binvox
from mpi4py import MPI

data_folder = '../../data/raw/ENZYME_pdbs/'
res_i = None
nb_rot = 1
chain = 'A'

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
                        entries.append([data_folder+class_fn+'/'+pdb_fn, i, data_folder+save_fn+'/'+pdb_fn.split('.')[0]+'_'+chain+'-r'+str(i)+'.png'])
        entries = np.array(entries)
        np.random.seed(9999)
        np.random.shuffle(entries)
    else: entries = None
    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]

    # Intialize Data Generator
    pdb_datagen = PDB_DataGenerator(size=64, center=[0,0,0], resolution=1.0, nb_rots=nb_rot, map_to_2d=True,
                                    channels=[hydrophobic_res, polar_res, charged_res, alpha_carbons, beta_carbons])
    print(len(entries))

    for i in range(len(entries)):
        # Entry Data
        pdb_path = entries[i][0]
        rot = int(entries[i][1])
        save_path = entries[i][2]

        # Generate and Save Data
        #t = time()
        pdb_array = pdb_datagen.generate_data(pdb_path, chain, res_i, rot)

        #scipy.misc.toimage(pdb_array, cmin=0, cmax=255).save(save_path)
        #print("Processing Time:", time()-t)

        '''
        t = time()
        array = imread(save_path)
        array = array[:,:,0] + (array[:,:,1] * 2**8) + (array[:,:,2] * 2**16)
        array = np.expand_dims(array.astype('>i8'), axis=-1)
        nb_chans = len(bin(np.max(array))[2:])
        array = np.unpackbits(array.view('uint8'),axis=-1)[:,:,-nb_chans:]
        array = np.flip(array, axis=-1)
        array = array * 255

        resized_array = []
        for i in range(nb_chans):
            temp = imresize(array[:,:,i], (128, 128), interp='bicubic')
            min_ = np.min(temp)
            temp = (temp - min_) / (np.max(temp) - min_)
            resized_array.append(temp)
        resized_array = np.transpose(np.array(resized_array), (1,2,0))

        print("Load Time:", time()-t)
        plt.imshow(resized_array[:,:,2:5])
        plt.show()
        exit()
        '''

    print("Done")
