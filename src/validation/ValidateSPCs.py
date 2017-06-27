'''
ValidateSPCs.py
Updated: 06/27/17

README:

The following script is used to run validation experiments on various 3D to 2D
spacefilling curve transpositions.

Global variables used to run experiments are defined under #- Global Variables.

'''
import os
import numpy as np
import matplotlib.pyplot as plt

#- Global Variables

#- Verbose Settings
debug = True

################################################################################

def calc_dist_matrix(points):
    '''
    Method returns distance matrix for given list of points.

    '''
    numPoints = len(points)
    distMat = sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1))
    return distMat.reshape((numPoints,numPoints))

def mse(imageA, imageB):
    '''
    Method returns the mean squared error between the two input images.

    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

if __name__ == '__main__':

    # Distance Comparison
    if debug: print("Comparison...")
    c3d_dist_matrix = calc_dist_matrix(curve_3d)
    c3d_dist_matrix = c3d_dist_matrix / np.amax(c3d_dist_matrix)
    #plt.imshow(c3d_dist_matrix, clim=(0.0, 1.0))
    #plt.colorbar()
    #plt.show()

    c2d_dist_matrix = calc_dist_matrix(curve_2d)
    c2d_dist_matrix = c2d_dist_matrix / np.amax(c2d_dist_matrix)
    #plt.imshow(c2d_dist_matrix, clim=(0.0, 1.0))
    #plt.colorbar()
    #plt.show()

    print 'MSE Between Curves:', mse(c3d_dist_matrix, c2d_dist_matrix)
    print 'SSIM Between Curves:', ssim(c3d_dist_matrix, c2d_dist_matrix)
