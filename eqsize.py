import numpy as np
import scipy.linalg

def eqsize(*args):
	m = 0
	for a in range(0, len(args)):
		p1 = args[a]

		for i in range(0, p1.size):
			m = maximum(m,(p1[i].monomials).shape[0])

	for a in range(len(args),-1,-1):
		p1 = args[a]
		for i in range(0,p1.size):
			if (p1[i].monomials).shape[0] < m:
				p1[i].monomials[m,:] = 0

		varargout[a] = p1

	return varargout

print __name__


