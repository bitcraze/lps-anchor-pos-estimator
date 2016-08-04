import multipol
import numpy as np

def toa_calc_d_from_xy(x,y):

	dimx = x.shape
	x_dim = dim[0]
	m = dim[1]
	dimy = y.shape
	y_dim = dim[0]
	n = dim[1]

	if x_dim > y_dim:

		y[(y_dim+1):x_dim,] = zeros(x_dim - y_dim, n)
	elif y_dim > x_dim:

		x[(x_dim+1):y_dim,] = zeros(y_dim - x_dim, n)

	d = sqrt(((np.kron(ones(1,n),x) - np.kron(y,ones(1,m))) ** 2).sum(axis=0))
	d = np.asarray(d)
	d = np.reshape(d,(m,n), order = 'F')

	return d
