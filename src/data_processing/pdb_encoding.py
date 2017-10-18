'''
pdb_encoding.py
Updated:

'''
from PDB_DataGenerator import PDB_DataGenerator
from PDB_DataGenerator import hydrophobic_res, polar_res, charged_res, alpha_carbons, beta_carbons
import numpy as np
from time import time
import matplotlib.pyplot as plt
from binvox_io import write_binvox, read_binvox

################################################################################

if __name__ == '__main__':

    # Intialize Data Generator
    t = time()
    pdb_datagen = PDB_DataGenerator(size=64, center=[0,0,0], resolution=1.0, nb_rots=10, map_to_2d=True,
                                    channels=[hydrophobic_res, polar_res, charged_res, alpha_carbons, beta_carbons])
    print('Init Time:',time() - t)

    # Generate Data
    t = time()
    pdb_rep = pdb_datagen.generate_data('0c1_t00000010_0001163.pdb', 'A',2)
    print('Gen Time:',time() - t)

    # Save Data
    t = time()
    write_binvox('test.binvox', np.transpose(pdb_rep,(2,0,1)))
    print('Gen Time:',time() - t)

    plt.imshow(pdb_rep[:,:,:3])
    plt.show()
