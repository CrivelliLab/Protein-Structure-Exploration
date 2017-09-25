'''
gen_rand_rot.py
Updated: 09/25/17

A script which will generate an arbitrary number of randomly-sampled points on
an n-sphere, avoiding polar clustering.

'''
import os
import numpy as np

num_points = 512
seed = 9283764

################################################################################

def gen_hypersphere_points(num_points, dimensions, seed, viz = False):
    '''
    Generates 3 vectors consisting of independent random samples from 3
    different gaussian distributions. These vectors are then normalized to
    match points lying on the surface of a unit sphere.

    INPUTS:
        npoints: integer number of points to generate.
        dims: integer number of dimensions for the hypersphere.
        viz: boolean True / False, whether to vizualize the resulting points on
            the given hypersphere.

    RETURNS:
        A Nx3 dimensional numpy array, where N is the number of points
        generated and 3 corresponds to the x, y, z coordinates of those points.

    For more information see this link, the last example of which inspired this
    approach:
        http://mathworld.wolfram.com/SpherePointPicking.html

    '''
    # For consistant random coordinate generation.
    np.random.seed(seed)

    # Sample rotations
    vector = np.random.randn(num_points, dimensions)
    vector /= np.linalg.norm(vector, axis=0)
    xi, yi, zi = vector

    # Combine the three Nx1 coordinate arrays into one Nx3 array describing x, y ,
    # z points.
    coordinate_arry = np.stack((xi, yi, zi), axis=-1)

    return coordinate_arry


if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    rot_file = '../../data/misc/rot_' + str(num_points) + '_' + str(seed)

    # Generate Random Rotations
    coordinates = gen_hypersphere_points(num_points, 3, seed, viz)
    np.save(rot_file, coordinates)
