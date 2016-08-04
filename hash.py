import multipol
import numpy as np
import scipy.linalg

def hash(p):

	m = (abs(p.coeffs)).max(0)
	A = concatenate((p.coeffs/m * 65535, p.monomials))

	B = concatenate((abs(A), np.sign(A)+1))
	h = chr(concatenate((np.log(m+1), B[:])))

	return h

