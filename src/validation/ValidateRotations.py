'''
ValidateRotations.py
Updated: 06/27/17

README:

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.measure import compare_ssim as ssim

#- Global Variables
folder = 'RAS-MD512-HH'

#- Verbose Settings
debug = True

def compare_images(encoded_folder):
    '''
    '''

    # Read PDB IMG File Names
    pdb_imgs = []
    for line in sorted(os.listdir('../../data/final/' + encoded_folder + '/')):
        if line.endswith('-0.png'):
            temp = []
            for i in range(512):
                temp.append(line.split('-')[0] + '-' + str(i) + '.png')
            pdb_imgs.append(temp)
    pdb_imgs = np.array(pdb_imgs)

    if debug: print "Comparing Encoded Images From", encoded_folder, '...'

    # Load PDB Images
    avg = 0
    for i in range(len(pdb_imgs)):
        img1 = misc.imread('../../data/final/' + encoded_folder + '/' + pdb_imgs[i][0])
        sims = 0
        for j in range(len(pdb_imgs[i])):
            #if debug: print "Comparing", pdb_imgs[i][0], "and", pdb_imgs[i][j]
            img2 = misc.imread('../../data/final/' + encoded_folder + '/' + pdb_imgs[i][j])
            mserror = mse(img1, img2)
            sim = ssim(img1, img2, multichannel=True)
            #if debug: print "SSIM", sim
            sims += sim
        sims = sims / len(pdb_imgs[i])
        avg += sims
    avg = avg / len(pdb_imgs)
    if debug: print 'Average Sim:', avg


def mse(imageA, imageB):
    '''
    Method returns the mean squared error between the two input images.

    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

if __name__ == '__main__':
    compare_images(folder)
