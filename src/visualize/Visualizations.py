'''
Visualizations.py
Updated: 06/20/17

'''

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

if __name__ == '__main__':
    pass
