'''
REVISED: 6 September 2017
A script which will generate an arbitrary number of randomly-sampled points on
an n-sphere, avoiding polar clustering. 
'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def gen_hypersphere_points(num_points, dimensions, viz = False):
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
    np.random.seed(9283764) # For consistant random coordinate generation.

    def sample_unit_sphere(npoints, dims):
        vector = np.random.randn(dims, npoints)
        vector /= np.linalg.norm(vector, axis=0)
        return vector

    def viz_points(coord_array):
        '''
        Takes in a numpy array where each row is the x,y,z coordinates of a point
        lying along a hypersphere. Vizualizes these points on the sphere's surface
        using matplotlib. Also makes use of x, y, z defined below in parent fn
        body. 
        '''
        # Plot all sampled points in 3d via matplotlib.
        x_coords, y_coords, z_coords = np.hsplit(coord_array, 3)
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
        ax.plot_wireframe(x, y, z, color='g', rstride=1, cstride=1)
        ax.scatter(x_coords, y_coords, z_coords, s=50, c='r', zorder=10)
        plt.show()
    
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    xi, yi, zi = sample_unit_sphere(num_points, dimensions)

    # Combine the three Nx1 coordinate arrays into one Nx3 array describing x, y ,
    # z points. 
    coordinate_arry = np.stack((xi, yi, zi), axis=-1)

    if viz:
        viz_points(coordinate_arry)
    
    return coordinate_arry


if __name__ == '__main__':
    coordinates = gen_hypersphere_points(512, 3, True)
