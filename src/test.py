import numpy as np
from mayavi import mlab

def sphere_mesh_coords():
    phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

if __name__ == '__main__':
    mlab.clf()
    x, y, z = sphere_mesh_coords()
    xx = x + 1
    yy = y + 1
    zz = z + 1
    x = np.concatenate([x,xx], 1)
    y = np.concatenate([y,yy], 1)
    z = np.concatenate([z,zz], 1)

    mlab.mesh(x, y, z)
    #mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
    mlab.show()
