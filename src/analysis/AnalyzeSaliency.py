'''
AnalyzeSaliency.py
Updated: 07/29/17

'''
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import jet
import matplotlib.ticker as ticker
from tvtk.api import tvtk
from tvtk.common import configure_input_data
from tqdm import tqdm
import numpy as np
import vtk
from scipy import misc
from sklearn.cluster import DBSCAN
from prody import parsePDB, moveAtoms, confProDy
confProDy(verbosity='none')

#- Global Variables
pdb_id = '1aa9A'
rot_id = 100
curve_3d = 'hilbert_3d_6.npy'
curve_2d = 'hilbert_2d_9.npy'
processed_file = 'HRASBOUNDED0%64_t45.npy'
pdb_folder = 'HRASBOUNDED0%64'
threshold = 0.7
nn = 6
eps = 2.5
samples = 10

#- Verbose Settings
debug = True

################################################################################

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
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[1], c2d[0]]

    return array_3d

def apply_rotation(pdb_data, rotation):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    Param:
        pdb_data - np.array ; multichanneled pdb atom coordinates
        rotation - np.array ; rotation matrix

    Return:
        pdb_data - np.array ; rotated multichanneled pdb atom coordinates

    '''

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    return rotated_pdb_data

def display_3d_model(pdb_data, pointcloud=None, centroids=None):
    '''
    Method renders space-filling atomic model of PDB data.

    Param:
        pdb_data - np.array ; mulitchanneled pdb atom coordinates
        skeletal - boolean ; if true shows model without radial information
        attenmap - np.array

    '''

    # Dislay 3D Mesh Rendering
    v = mlab.figure(bgcolor=(1.0,1.0,1.0))

    if pdb_data is not None:
        # Color Mapping
        n = len(pdb_data)
        cm = [jet(float(i)/n)[:3] for i in range(n)]

        for j in range(len(pdb_data)):
            c = cm[j]

            # Coordinate, Radius Information
            r = pdb_data[j][:,0].astype('float')
            x = pdb_data[j][:,1].astype('float')
            y = pdb_data[j][:,2].astype('float')
            z = pdb_data[j][:,3].astype('float')

            # Generate Mesh For Protein
            append_filter = vtk.vtkAppendPolyData()
            for i in range(len(pdb_data[j])):
                input1 = vtk.vtkPolyData()
                sphere_source = vtk.vtkSphereSource()
                sphere_source.SetCenter(x[i],y[i],z[i])
                sphere_source.SetRadius(r[i])
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
            p = tvtk.Property(opacity=1.0, color=c)
            sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
            v.scene.add_actor(sphere_actor)

    if pointcloud is not None:

        # Generate Voxels For Protein
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(xx)):
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
        p = tvtk.Property(opacity=1.0, color=(0.9,0.9,0.9))
        cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
        v.scene.add_actor(cube_actor)

    if centroids is not None:

        # Generate Mesh For Protein
        append_filter = vtk.vtkAppendPolyData()
        for i in range(len(centroids)):
            input1 = vtk.vtkPolyData()
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(centroids[i][0],centroids[i][1],centroids[i][2])
            sphere_source.SetRadius(2.0)
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
        p = tvtk.Property(opacity=1.0, color=(1.0,0.0,0.0))
        sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
        v.scene.add_actor(sphere_actor)

    mlab.show()

def closest_nodes(node, nodes, nb_hits):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    hits = []
    for i in range(nb_hits):
        x = np.argmin(dist_2)
        dist_2[x] = 1000
        if nodes[x] is [-1,-1,-1]: print "eror";continue
        hits.append(x)
    return hits

if __name__ == '__main__':

    # File Paths
    curve_2d = '../../data/raw/SFC/'+ curve_2d
    curve_3d = '../../data/raw/SFC/'+ curve_3d
    processed_file = '../../data/interim/' + processed_file
    pdb = pdb_id + '-r' + str(rot_id) + '.png'
    pdb_folder = '../../data/raw/PDB/' + pdb_folder + '/'

    # Load Curves
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # Load Protein Data
    data = np.load(processed_file)
    rot = data[1][rot_id]
    pdb_data = None
    for d in data[0]:
        if d[0] == pdb_id: pdb_data = d[1]
    pdb_data = apply_rotation(pdb_data, rot)

    # Calculate Protein Diameter
    dia = 0
    for channel in pdb_data:
        temp = np.amax(np.abs(channel[:, 1:])) + 2
        if temp > dia: dia = temp

    # Load Saliency Map
    attenmap_2d = None
    attenmap_3d = None
    attenmap_2d = misc.imread('../../data/analysis/' + pdb)
    attenmap_2d = attenmap_2d.astype('float')/255.0
    attenmap_2d[attenmap_2d < threshold] = 0
    attenmap_3d = map_2d_to_3d(attenmap_2d, curve_3d, curve_2d)

    # Generate Saliency Pointcloud
    xx, yy, zz = np.where(attenmap_3d > 0.0)
    ww = [attenmap_3d[xx[i],yy[i],zz[i]] for i in range(len(xx))]
    xx = (xx * (dia*2)/len(attenmap_3d[0])) - dia
    yy = (yy * (dia*2)/len(attenmap_3d[0])) - dia
    zz = (zz * (dia*2)/len(attenmap_3d[0])) - dia
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

    # Parse PDB
    molecule = parsePDB(pdb_folder+pdb_id[:-1]+'.pdb.gz').select('protein')
    molecule = molecule.select('chain '+pdb_id[-1])
    moveAtoms(molecule, to=np.zeros(3))
    hv = molecule.getHierView()[pdb_id[-1]]

    # Gather Residuel Information
    res_centroids = []
    residuels = []
    for i, residue in enumerate(hv.iterResidues()):
        res = hv[i]
        if res is None:
            res_centroids.append([-1,-1,-1])
            residuels.append("UNKNOWN")
            continue
        res_coord = res.getCoords()
        x_mean = np.mean(res_coord[:,0])
        y_mean = np.mean(res_coord[:,1])
        z_mean = np.mean(res_coord[:,2])
        res_centroids.append([x_mean, y_mean, z_mean])
        residuels.append(residue)
    res_centroids = np.array(res_centroids)

    # Calculate Nearest Neighbors
    cluster_hits = []
    for i in range(len(cluster_centroids)):
        hits = closest_nodes(cluster_centroids[i], res_centroids, nn)
        cluster_hits.append(hits)
    cluster_hits = np.array(cluster_hits)

    #display_3d_model(None, saliency_pointcloud, cluster_centroids)

    # Generate Focus Map
    mat = np.zeros((len(cluster_hits), len(residuels)))
    for i in range(len(cluster_hits)):
        for j in range(len(cluster_hits[i])):
            mat[i][cluster_hits[i][j]] = len(cluster_hits[i]) - j

    # Dissplay Focus Map
    sections = int(len(residuels)/60) + (len(residuels)%60 > 0)
    f, axarray = plt.subplots(sections, figsize=(9,9))
    for i in range(sections):
        ax = axarray[i]
        cax = ax.matshow(mat[:,0+(60*i):60+(60*i)], cmap=plt.cm.Blues)

        #if i+1 == sections:


        ax.set_xticklabels([''] + [str(j) for j in range(0+(60*i),60+(60*i),5)])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_ticks_position('bottom')

        if i == 0:
            ax.set_title(pdb[:-4] + ", N-Nearest Neighbor Residues of Saliency Clusters; n="+str(nn) +"; eps="+str(eps)+"; min_sample="+str(samples))
    plt.ylabel('Saliency Cluster')
    plt.xlabel('Residues')
    f.subplots_adjust(bottom=0.2)
    cbar_ax = f.add_axes([0.25, -0.55, 0.5, 0.7])
    cbar = f.colorbar(cax, ticks=[0, nn/2, nn],pad=0.5,location='top')
    cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])



    plt.show()
