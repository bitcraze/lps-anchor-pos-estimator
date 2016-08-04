import numpy as np
import scipy.linalg

def nvars(p):
	n = zeros(p.shape)
	for i in range(p.size):
		n[i] = (p[i].monomials).shape[0]

	return n