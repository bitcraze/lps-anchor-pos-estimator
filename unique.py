import numpy as np
from eqsize import *
import scipy.linalg

def unique(eq):
	eq = eqsize(eq)
	c1 = [None] * eq.shape
	for i in range(0,eq.size):
		c1.append[i] = hash(eq[i])

	c1 = np.asarray(c1)

	if c1.ndim == 1:
		_, ia, ic = np.unique(c1, return_index=True, return_inverse=True)
		ia = (ia[:,]).conj().T
		ic = (ic[:,]).conj().T
		u = eq[ia]

	else:
		 a = c1
		 b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
		 _,ia,ic = np.unique(b,return_index=True,return_inverse=True)



	return u, ia, ic