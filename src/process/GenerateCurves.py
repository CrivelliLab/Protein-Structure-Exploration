'''
GenerateCurves.py
Updated: 6/26/17

README:

The following script is used to generate 3D and 2D spacefilling curves.

Global variables used to generate the curves are defined under #- Global Variables.
gen_3d and gen_2d define the 3D and 2D curve which will be generated.
order_3d and order_2d define what order curve will be generated.

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
Script prints out pairs of orders for 3D and 2D curves that will allow for this.

Generated curve array files will be save under data/source/SFC/ with the following
naming convention:

<curve><order>.npy

'''
import numpy as np
from scipy import reshape, sqrt, identity
from numpy.matlib import repmat, repeat
import matplotlib.pyplot as plt

#- Global Variables
gen_3d = 'hilbert_3D'
gen_2d = 'fold_2D'
order_3d = 4
order_2d = 6

#- Verbose Settings
debug = True

### Z-Curves ###################################################################

def zcurve_3D(n):
    '''
    Method generates 3D z-order curve of order n.

    '''
    n = pow(pow(2, n), 3)
    curve = []
    for i in range(n):
        x, y, z = deinterleave3(i)
        curve.append([x, y, z])
    return np.array(curve)

def zcurve_2D(n):
    '''
    Method generates 2D z-order curve of order n.

    '''
    n = pow(pow(2, n), 2)
    curve = []
    for i in range(n):
        x, y = deinterleave2(i)
        curve.append([x, y])
    return np.array(curve)

def part1by1(n):
        n&= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

def unpart1by1(n):
        n&= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0f0f0f0f
        n = (n ^ (n >> 4)) & 0x00ff00ff
        n = (n ^ (n >> 8)) & 0x0000ffff
        return n

def interleave2(x, y):
        return part1by1(x) | (part1by1(y) << 1)

def deinterleave2(n):
        return unpart1by1(n), unpart1by1(n >> 1)

def part1by2(n):
        n&= 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n <<  8)) & 0x0300f00f
        n = (n ^ (n <<  4)) & 0x030c30c3
        n = (n ^ (n <<  2)) & 0x09249249
        return n

def unpart1by2(n):
        n&= 0x09249249
        n = (n ^ (n >>  2)) & 0x030c30c3
        n = (n ^ (n >>  4)) & 0x0300f00f
        n = (n ^ (n >>  8)) & 0xff0000ff
        n = (n ^ (n >> 16)) & 0x000003ff
        return n


def interleave3(x, y, z):
        return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)

def deinterleave3(n):
        return unpart1by2(n), unpart1by2(n >> 1), unpart1by2(n >> 2)

### Hilbert Curve ##############################################################

def hilbert_3D(n):
	'''
    Method generates 3D hilbert curve of order n.

	'''
	curve = [(0,0,0)]
	generate3d(curve, curve[-1],n, 0, 1, 2, 3, 4, 5, 6, 7 )
	return np.array(curve)

def hilbert_2D(n):
	'''
    Method generates 2D hilbert curve of order n.

	'''
	curve = []
	generate2d(curve, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, n)
	curve = np.array(curve) * pow(2, n)
	return curve.astype("int")

def add_to_list(l, v):
	l.append( (l[-1][0]+v[0], l[-1][1]+v[1], l[-1][2]+v[2]) )

def select(curve, v, i, w0, w1, w2, w3, w4, w5, w6, w7):
	term ={	0 : w0, 1 : w1, 2 : w2, 3 : w3, 4 : w4, 5 : w5, 6 : w6, 7 : w7}
	st = term[i]
	nd = term[i+1]
	vec = ( v[nd][0]-v[st][0], v[nd][1]-v[st][1], v[nd][2]-v[st][2] )
	add_to_list(curve, vec)
	return i+1

def generate3d(curve, sr, n, _0, _1, _2, _3, _4, _5, _6, _7):
	v = [	( 0, 0, 0 ), ( 0, 0, 1 ), ( 0, 1, 1 ), ( 0, 1, 0 ),
			( 1, 1, 0 ), ( 1, 1, 1 ), ( 1, 0, 1 ), ( 1, 0, 0 )]

	if n == 0:return

	i = 0
	generate3d(curve, curve[-1], n-1, _0, _3, _4, _7, _6, _5, _2, _1)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _0, _7, _6, _1, _2, _5, _4, _3)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _0, _7, _6, _1, _2, _5, _4, _3)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _2, _3, _0, _1, _6, _7, _4, _5)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _2, _3, _0, _1, _6, _7, _4, _5)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _4, _3, _2, _5, _6, _1, _0, _7)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _4, _3, _2, _5, _6, _1, _0, _7)
	i = select(curve,v,i, _0, _1, _2, _3, _4, _5, _6, _7 )
	generate3d(curve, curve[-1], n-1, _6, _5, _2, _1, _0, _3, _4, _7)

def generate2d(curve, x0, y0, xi, xj, yi, yj, n):
	if n <= 0:
		X = x0 + (xi + yi)/2
		Y = y0 + (xj + yj)/2
		curve.append((X,Y))
	else:
		generate2d(curve, x0, y0, yi/2, yj/2, xi/2, xj/2, n - 1)
		generate2d(curve, x0 + xi/2, y0 + xj/2, xi/2, xj/2, yi/2, yj/2, n - 1)
		generate2d(curve, x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, n - 1)
		generate2d(curve, x0 + xi/2 + yi, y0 + xj/2 + yj, -yi/2,-yj/2,-xi/2,-xj/2, n - 1)

### 2D Folding ##########################################################################

def fold_2D(n):
    '''
    '''
    s = pow(2, n)
    curve = []
    for i in range(s):
        for j in range(s):
            if i % 2 == 0: curve.append([j, i])
            else: curve.append([s-j-1, i])
    curve = np.array(curve)
    return curve

################################################################################

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
            print "Total Number of Pixels:", x, '\n'

if __name__ == '__main__':

    # File Paths
    curves_folder = '../../data/source/SFC/'

    # Display Possible Order Pairings
    display_spacefilling_dim()

    # Generate Space Filling Curves
    if debug: print("Generating 3D Curve...")
    curve_3d = globals()[gen_3d](order_3d)
    np.save(curves_folder+gen_3d+str(order_3d), curve_3d)

    if debug: print("Generating 2D Curve...")
    curve_2d = globals()[gen_2d](order_2d)
    np.save(curves_folder+gen_2d+str(order_2d), curve_2d)
