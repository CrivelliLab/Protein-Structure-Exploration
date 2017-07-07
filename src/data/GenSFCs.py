'''
GenSFCs.py
Updated: 7/8/17

README:

The following script is used to generate 3D and 2D spacefilling curves.

Global variables used to generate the curves are defined under #- Global Variables.
'gen' defines the curve which will be generated. 'order' defines what order curve will be generated.

Currently the following spacefilling curves are implemented:

- 3D Z-order : 'zcurve_3D'
- 2D Z-order : 'zcurve_2D'
- 3D Hilbert : 'hilbert_3D'
- 2D Hilbert : 'hilbert_2D'
- 2D Naive Folding : 'fold_2D'

The 2^order determines the length along an axis. For example:

- order 4 2D curve results in 16 X 16 array.
- order 4 3D curve results in 16 X 16 X 16 array.

Note: For 3D to 2D transposition, length of 3D and 2D curves must be the same.
Script prints out pairs of orders for 3D and 2D curves that will map properly.

Generated curve array files will be save under data/source/SFC/ with the following
naming convention:

<curve><order>.npy

'''
import numpy as np

#- Global Variables
gen = 'zcurve_3D' # Name of function used to generate SFC
order = 4

#- Verbose Settings
debug = True

################################################################################

def zcurve_3D(order):
    '''
    Method generates 3D z-order curve of desired order.

    '''
    z_curve = []
    for i in range(pow(pow(2, order),3)):
        x = i
        x &= 0x09249249
        x = (x ^ (x >>  2)) & 0x030c30c3
        x = (x ^ (x >>  4)) & 0x0300f00f
        x = (x ^ (x >>  8)) & 0xff0000ff
        x = (x ^ (x >> 16)) & 0x000003ff

        y = i >> 1
        y &= 0x09249249
        y = (y ^ (y >>  2)) & 0x030c30c3
        y = (y ^ (y >>  4)) & 0x0300f00f
        y = (y ^ (y >>  8)) & 0xff0000ff
        y = (y ^ (y >> 16)) & 0x000003ff

        z = i >> 2
        z &= 0x09249249
        z = (z ^ (z >>  2)) & 0x030c30c3
        z = (z ^ (z >>  4)) & 0x0300f00f
        z = (z ^ (z >>  8)) & 0xff0000ff
        z = (z ^ (z >> 16)) & 0x000003ff

        z_curve.append([x, y, z])

    return np.array(z_curve)

def zcurve_2D(order):
    '''
    Method generates 2D z-order curve of desired order.

    '''
    z_curve = []
    for i in range(pow(pow(2, order),2)):
        x = i
        x&= 0x55555555
        x = (x ^ (x >> 1)) & 0x33333333
        x = (x ^ (x >> 2)) & 0x0f0f0f0f
        x = (x ^ (x >> 4)) & 0x00ff00ff
        x = (x ^ (x >> 8)) & 0x0000ffff

        y = i >> 1
        y&= 0x55555555
        y = (y ^ (y >> 1)) & 0x33333333
        y = (y ^ (y >> 2)) & 0x0f0f0f0f
        y = (y ^ (y >> 4)) & 0x00ff00ff
        y = (y ^ (y >> 8)) & 0x0000ffff

        z_curve.append([x, y])

    return np.array(z_curve)

def hilbert_3D(order):
    '''
    Method generates 3D hilbert curve of desired order.

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

    return np.array(hilbert_curve)

def hilbert_2D(order):
    '''
    Method generates 2D hilbert curve of desired order.

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

    return np.array(hilbert_curve)

def fold_2D(order):
    '''
    Method generates 2D square folding curve of desired order.

    '''
    s = pow(2, order)
    curve = []
    for i in range(s):
        for j in range(s):
            if i % 2 == 0: curve.append([j, i])
            else: curve.append([s-j-1, i])
    curve = np.array(curve)
    return curve

def display_spacefilling_dim():
    '''
    Method displays dimensions of various order space filling curves.

    '''
    # Calculating SFC 3D to 2D Mapping Pair Dimensions
    for i in range(64):
        x = pow(2,i)
        sq = np.sqrt(x)
        cb = np.cbrt(x)
        if sq %1.0 == 0.0 and cb %1.0 == 0.0:
            print "\nSpace Filling 2D Curve:", int(sq), 'x', int(sq), ', order-', np.log2(sq)
            print "Space Filling 3D Curve:", int(cb), 'x', int(cb), 'x', int(cb), ', order-', np.log2(cb)
            print "Total Number of Pixels:", x, '\n'

if __name__ == '__main__':

    # File Paths
    curves_folder = '../../data/raw/SFC/'

    # Display Possible Order Pairings
    if debug: print("Displaying Possible Mapping Combinations...")
    display_spacefilling_dim()

    # Generate Space Filling Curve
    if debug: print("Generating Curve...")
    curve = globals()[gen](order)
    np.save(curves_folder+gen+str(order), curve)
