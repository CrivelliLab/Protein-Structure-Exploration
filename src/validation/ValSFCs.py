'''
ValSFCs.py
Updated: 07/12/17
[PASSING]

README:

The following script is used to run validation experiments on various 3D to 2D
spacefilling curve transpositions.

Global variables used to run experiments are defined under #- Global Variables.
'curve_3d' and 'curve_2d' defines the curves that will be compared.
Curves must be curve array files found under data/raw/SFC/ .

'normalize' defines whether generated distance matriceis of curves are normalized
before difference metrics are caluclated.

The following difference metrics will be calculated from the distance matricies
of the 3D and 2D curves:

- Mean Squared Error
- Structural Similarity Index

'''
import numpy as np
from skimage.measure import compare_ssim as ssim
from numpy.matlib import repmat, repeat

#- Global Variables
curve_3d = 'hilbert_3D4.npy'
curve_2d = 'hilbert_2D6.npy'
normalize = False

#- Verbose Settings
debug = True

################################################################################

def calc_dist_matrix(points):
    '''
    Method returns distance matrix for given list of points.

    Param:
        points - np.array ; list of coordinates

    '''
    numPoints = len(points)
    distMat = np.sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1))
    return distMat.reshape((numPoints,numPoints))

def mse(imageA, imageB):
    '''
    Method returns the mean squared error between the two input images.

    Param:
        imageA - np.array
        imageB - np.array

    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

if __name__ == '__main__':

    # File Paths
    path_to_project = '../../'
    curve_2d = path_to_project + 'data/raw/SFC/'+ curve_2d
    curve_3d = path_to_project + 'data/raw/SFC/'+ curve_3d

    # Load Curves
    if debug: print("Loading Curves...")
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # Distance Comparison
    if debug: print("Generating Distance Matricies...")
    c3d_dist_matrix = calc_dist_matrix(curve_3d)
    c2d_dist_matrix = calc_dist_matrix(curve_2d)

    # Normalize
    if normalize:
        if debug: print("Normalizing...")
        c3d_dist_matrix = c3d_dist_matrix / np.amax(c3d_dist_matrix)
        c2d_dist_matrix = c2d_dist_matrix / np.amax(c2d_dist_matrix)

    # Display Results
    if debug: print("Calculating Difference Metrics...")
    print 'MSE Between Curves:', mse(c3d_dist_matrix, c2d_dist_matrix)
    print 'SSIM Between Curves:', ssim(c3d_dist_matrix, c2d_dist_matrix)
