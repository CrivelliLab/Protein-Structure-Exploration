'''
HilbertCurves.py
Last Updated: 5/7/2017

'''
import numpy as np

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

def gen_hilbert_3D(n):
	curve = [(0,0,0)]
	generate3d(curve, curve[-1],n, 0, 1, 2, 3, 4, 5, 6, 7 )
	return np.array(curve)

def gen_hilbert_2D(n):
	curve = []
	generate2d(curve, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, n)
	curve = np.array(curve) * pow(2, n)
	return curve.astype("int")
