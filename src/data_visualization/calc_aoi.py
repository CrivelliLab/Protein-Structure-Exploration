'''
calculate_aoi.py
Updated: 2/12/18

This script is used to calculate areas of interest using saliency maps generated
from trained network parameters.

'''
from mayavi import mlab
import os
import vtk
import numpy as np
from tvtk.api import tvtk
from matplotlib.cm import *
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
from tvtk.common import configure_input_data

# Data Parameters
data_path = '../../data/KrasHras/Hras/1aa9_A'
threshold = 0.7
eps = 2.0
samples = 10
resolution = 1.0

################################################################################

seed = 1234

def hilbert_3d(order):
    '''
    Method generates 3D hilbert curve of desired order.

    Param:
        order - int ; order of curve

    Returns:
        np.array ; list of (x, y, z) coordinates of curve

    '''

    def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
        if order == 0:
            xx = x + (xi + yi + zi)/3
            yy = y + (xj + yj + zj)/3
            zz = z + (xk + yk + zk)/3
            array.append((xx, yy, zz))
        else:
            gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

            gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                       yi/2, yj/2, yk/2, array)
            gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                       xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
            gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                       -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
            gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                       -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
            gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                       -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
            gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                       -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
            gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                       -zk/2, -xi/2, -xj/2, -xk/2, array)

    n = pow(2, order)
    hilbert_curve = []
    gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

    return np.array(hilbert_curve).astype('int')

def hilbert_2d(order):
    '''
    Method generates 2D hilbert curve of desired order.

    Param:
        order - int ; order of curve

    Returns:
        np.array ; list of (x, y) coordinates of curve

    '''
    def gen_2d(order, x, y, xi, xj, yi, yj, array):
        if order == 0:
            xx = x + (xi + yi)/2
            yy = y + (xj + yj)/2
            array.append((xx, yy))
        else:
            gen_2d(order-1, x, y, yi/2, yj/2, xi/2, xj/2, array)
            gen_2d(order-1, x + xi/2, y + xj/2, xi/2, xj/2, yi/2, yj/2, array)
            gen_2d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, array)
            gen_2d(order-1, x + xi/2 + yi, y + xj/2 + yj, -yi/2,-yj/2,-xi/2,-xj/2, array)

    n = pow(2, order)
    hilbert_curve = []
    gen_2d(order, 0, 0, n, 0, 0, n, hilbert_curve)

    return np.array(hilbert_curve).astype('int')

def map_2d_to_3d(array_2d, curve_3d, curve_2d):
    '''
    Method proceses 2D array and encodes into 3D using SFC.
    Param:
        array_2d - np.array
        curve_3d - np.array
        curve_2d - np.array
    Return:
        array_3d - np.array

    '''
    s = int(np.cbrt(len(curve_3d)))
    array_3d = np.zeros([s,s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[0], c2d[1]]

    return array_3d

def display_3d_model(array_3d, pointcloud=None, centroids=None):
    '''
    Method renders space-filling atomic model of PDB data.
    Param:
        pdb_data - np.array ; mulitchanneled pdb atom coordinates
        skeletal - boolean ; if true shows model without radial information
        attenmap - np.array
    '''

    # Dislay 3D Mesh Rendering
    v = mlab.figure(bgcolor=(1.0,1.0,1.0))

    if array_3d is not None:

        # Color Mapping
        n = len(array_3d)
        cm = [winter(float(i)/n)[:3] for i in range(n)]

        for j in range(len(array_3d)):
            c = tuple(cm[j])

            # Coordinate Information
            xx, yy, zz = np.where(array_3d[j] > 0.0)
            xx = xx.astype('float') - ((resolution * array_3d.shape[1])/2.0)
            yy = yy.astype('float') - ((resolution * array_3d.shape[1])/2.0)
            zz = zz.astype('float') - ((resolution * array_3d.shape[1])/2.0)

            # Generate Voxels For Protein
            append_filter = vtk.vtkAppendPolyData()
            for i in range(len(xx)):
                input1 = vtk.vtkPolyData()
                voxel_source = vtk.vtkCubeSource()
                voxel_source.SetCenter(xx[i],yy[i],zz[i])
                voxel_source.SetXLength(1)
                voxel_source.SetYLength(1)
                voxel_source.SetZLength(1)
                voxel_source.Update()
                input1.ShallowCopy(voxel_source.GetOutput())
                append_filter.AddInputData(input1)
            append_filter.Update()

            #  Remove Any Duplicate Points.
            clean_filter = vtk.vtkCleanPolyData()
            clean_filter.SetInputConnection(append_filter.GetOutputPort())
            clean_filter.Update()

            # Render Voxels
            pd = tvtk.to_tvtk(clean_filter.GetOutput())
            cube_mapper = tvtk.PolyDataMapper()
            configure_input_data(cube_mapper, pd)
            p = tvtk.Property(opacity=1.0, color=c)
            cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
            v.scene.add_actor(cube_actor)

    if pointcloud is not None:

        # Generate Voxels For Pointcloud
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(pointcloud)):
            input1 = vtk.vtkPolyData()
            voxel_source = vtk.vtkCubeSource()
            voxel_source.SetCenter(pointcloud[i][0],pointcloud[i][1], pointcloud[i][2])
            voxel_source.SetXLength(1)
            voxel_source.SetYLength(1)
            voxel_source.SetZLength(1)
            voxel_source.Update()
            input1.ShallowCopy(voxel_source.GetOutput())
            append_filter.AddInputData(input1)
        append_filter.Update()

        #  Remove Any Duplicate Points.
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputConnection(append_filter.GetOutputPort())
        clean_filter.Update()

        # Render Voxels
        pd = tvtk.to_tvtk(clean_filter.GetOutput())
        cube_mapper = tvtk.PolyDataMapper()
        configure_input_data(cube_mapper, pd)
        p = tvtk.Property(opacity=1.0, color=(1.0,0.5,0.5))
        cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
        v.scene.add_actor(cube_actor)

    if centroids is not None:

        # Generate Mesh For Centroids
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(centroids)):
            input1 = vtk.vtkPolyData()
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(centroids[i][0],centroids[i][1],centroids[i][2])
            sphere_source.SetRadius(4.0)
            sphere_source.Update()
            input1.ShallowCopy(sphere_source.GetOutput())
            append_filter.AddInputData(input1)
        append_filter.Update()

        #  Remove Any Duplicate Points.
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputConnection(append_filter.GetOutputPort())
        clean_filter.Update()

        # Render Mesh
        pd = tvtk.to_tvtk(clean_filter.GetOutput())
        sphere_mapper = tvtk.PolyDataMapper()
        configure_input_data(sphere_mapper, pd)
        p = tvtk.Property(opacity=1.0, color=(1.0,0.0,1.0))
        sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
        v.scene.add_actor(sphere_actor)

    mlab.show()

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load data
    array_3d = np.load(data_path + '/' + data_path.split('/')[-1] + '-3d.npz')['arr_0'].astype('int')
    array_3d = np.transpose(array_3d,(3,0,1,2))
    atten_map = np.load(data_path + '/' + data_path.split('/')[-1] + '-2d_sal.npz')['arr_0']
    atten_map[atten_map < threshold] = 0

    # Generate Curves
    curve_3d = hilbert_3d(6)
    curve_2d = hilbert_2d(9)

    # Map 2D Saliency to 3D
    atten_map_3d = map_2d_to_3d(atten_map, curve_3d, curve_2d)

    # Get Point Cloud
    dia = (atten_map_3d.shape[0] * resolution) / 2.0
    xx, yy, zz = np.where(atten_map_3d > 0.0)
    ww = [atten_map_3d[xx[i],yy[i],zz[i]] for i in range(len(xx))]
    xx = (xx * (dia*2)/len(atten_map_3d[0])) - dia
    yy = (yy * (dia*2)/len(atten_map_3d[0])) - dia
    zz = (zz * (dia*2)/len(atten_map_3d[0])) - dia
    saliency_pointcloud = np.array([xx,yy,zz])
    saliency_pointcloud = np.transpose(saliency_pointcloud, (1,0))

     # Cluster Saliency Pointcloud, ignoring noise if present.
    db = DBSCAN(eps=eps, min_samples=samples).fit(saliency_pointcloud)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Calculate Centroid of Saliency Clusters
    cluster_centroids = []
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(n_clusters_):
        cluster_indexes = [j for j in range(len(labels)) if labels[j] == i]
        x_mean = np.mean(saliency_pointcloud[cluster_indexes,0])
        y_mean = np.mean(saliency_pointcloud[cluster_indexes,1])
        z_mean = np.mean(saliency_pointcloud[cluster_indexes,2])
        cluster_centroids.append([x_mean, y_mean, z_mean])
    cluster_centroids = np.array(cluster_centroids)

    # Save Centroid Data
    np.savez(data_path + '/' + data_path.split('/')[-1] + '-2d_ccs.npz', cluster_centroids)

    # Display
    display_3d_model(array_3d, saliency_pointcloud, None)
