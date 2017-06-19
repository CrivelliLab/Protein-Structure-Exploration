'''
VisualizationTools.py
Last Updated: 6/16/2017

'''
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data
from skimage.measure import compare_ssim
from scipy import ndimage

def display_3d_array(array_3d):
    '''
    Method displays 3d array.

    '''
    # Dislay 3D Voxel Rendering
    for i in range(len(array_3d)):
        if i == 1: c = (1, 0, 0)
        elif i == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)
        xx, yy, zz = np.where(array_3d[i] >= 1)
        mlab.points3d(xx, yy, zz, mode="cube", color=c)
    mlab.show()

def display_3d_mesh(pdb_data):
    '''
    '''
    # Dislay 3D Mesh Rendering
    v = mlab.figure()
    for j in range(len(pdb_data)):
        if j == 1: c = (1, 0, 0)
        elif j == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)

        # Coordinate, Radius Information
        x = pdb_data[j][:,3].astype('float')
        y = pdb_data[j][:,2].astype('float')
        z = pdb_data[j][:,1].astype('float')
        s = pdb_data[j][:,0].astype('float')

        # Generate Mesh For Protein
        for i in range(len(pdb_data[j])):
            sphere = tvtk.SphereSource(center=(x[i],y[i],z[i]), radius=s[i])
            sphere_mapper = tvtk.PolyDataMapper()
            configure_input_data(sphere_mapper, sphere.output)
            sphere.update()
            p = tvtk.Property(opacity=1.0, color=c)
            sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
            v.scene.add_actor(sphere_actor)

    mlab.show()

def display_3d_points(coords_3d):
    '''
    '''
    # Display 3D Mesh Rendering
    for i in range(len(coords_3d)):
        if i == 1: c = (1, 0, 0)
        elif i == 2: c = (0, 1, 0)
        else: c = (0, 0, 1)
        # Coordinate, Radius Information
        x = coords_3d[i][:,3].astype('float')
        y = coords_3d[i][:,2].astype('float')
        z = coords_3d[i][:,1].astype('float')

        mlab.points3d(x, y, z, mode="sphere", color=c, scale_factor=0.5)
    mlab.show()

def display_2d_array(array_2d):
    '''
    Method displays 2-d array.

    '''
    # Display 2D Plot
    plt.figure()
    plt.imshow(array_2d, interpolation="nearest")
    plt.show()

def display_spacefilling_dim():
    '''
    Method displays dimensions of various order space filling curves.

    '''
    # Calculating Hilbert 3D to 2D Conversion Dimensions
    for i in range(64):
        x = pow(2,i)
        sq = np.sqrt(x)
        cb = np.cbrt(x)
        if sq %1.0 == 0.0 and cb %1.0 == 0.0:
            print "\nSpace Filling 2D Curve:", int(sq), 'x', int(sq), ', order-', np.log2(sq)
            print "Space Filling 3D Curve:", int(cb), 'x', int(cb), 'x', int(cb), ', order-', np.log2(cb)
            print "Total Number of Pixels:", x

def display_hist(data, title=""):
    '''
    Method displays histogram of inputed data.

    '''
    plt.hist(data)
    plt.title(title)
    plt.show()

def display_image_similarty(image_1, image_2):
    '''
    '''
    err = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    err /= float(image_1.shape[0] * image_2.shape[1])
    print "MSE:", err

    sim = compare_ssim(image_1, image_2, multichannel=True)
    print "Structural Simularity:", sim

if __name__ == '__main__':
    from ZCurves import *
    from ProcessingTools import *
    array_3d = np.zeros((3,64,64,64))
    for i in range(3):
        for j in range(64):
            for x in range(64):
                for v in range(64):
                    if i == 2 and v > 0 and v < 64 and j >0 and j < 64 and x > 0 and x < 64:
                        array_3d[i][j][x][v] = 1
                    if i == 1 and v > 10 and v < 50 and j >10 and j < 50 and x > 10 and x < 64:
                        array_3d[i][j][x][v] = 1
                        #array_3d[2][j][x][v] = 0
                    if i == 0 and v > 21 and v < 41 and j >21 and j < 41 and x > 21 and x < 64:
                        array_3d[i][j][x][v] = 1
                        #array_3d[2][j][x][v] = 0
                        #array_3d[1][j][x][v] = 0

    curve_3d = gen_zcurve_3D(pow(64, 3))
    curve_2d = gen_zcurve_2D(pow(64, 3))
    # Process Channels
    encoded_2d = []
    for i in range(3):
        # Encode 3D Model with Space Filling Curve
        encoded_res_2d = encode_3d_pdb(array_3d[i], curve_3d, curve_2d, debug=False)
        encoded_2d.append(encoded_res_2d)

    # Transpose Array
    encoded_2d = np.array(encoded_2d)
    encoded_2d = np.transpose(encoded_2d, (2,1,0))

    display_3d_array(array_3d)
    display_2d_array(encoded_2d)
