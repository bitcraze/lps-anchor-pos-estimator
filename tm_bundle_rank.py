import multipol
import numpy as np
import scipy.linalg
from compactionmatrix import *
from scipy.sparse import csr_matrix
from calcresandjac import *


def toa_misstoa_bundle_rank(sol,d):
	r_c = (sol.Bhat).shape
	mm = r_c[0]
	nn = r_c[1]

	cl, dl = compactionmatrix(mm)
	cr, dr = compactionmatrix(nn)

	d2 = d ** 2

	d2shuffle = d2[sol.rows, sol.cols]
	ishuffle = so.inlmatrix[sol.rows, sol.cols]
	I, J = (ishuffle==1).nonzero()
	V = (np.array(range(1,size(I)+1))).conj().T

	d2vm = dshuffle[np.ravel_multi_index((I, J), dims = d2shuffle.shape, order = 'F')]

	ineededinBhat = ishuffle==1
	ineededinBhat[:,1] = ones(mm,1)
	ineededinBhat[1,:] = ones(1,nn)
	(I2, J2) = where(ineededinBhat!=0)

	ord1 = (J2==1).nonzero()
	ord2 = (I2==1 and J2>1).nonzero()
	ord3 = (I2>1 and J2>1).nonzero()
	I2 = I2[(concatenate((ord1,ord2,ord3)))]
	J2 = J2[(concatenate((ord1,ord2,ord3)))]
	nspecial = len(ord1) + len(ord2)
	nrest = len(ord3)
	I22 = I2[(nspecial+1):(nspecial+nrest)] - 1
	J22 = J2[(nspecial+1):(nspecial+nrest)] - 1
	V2 = range(1,len(I2)+1)
	ij2v = csr_matrix((V2, (I2,J2)),shape=(mm,nn)).toarray()

	II = V
	JJ = ij2v[np.ravel_multi_index((I,J), dims = ij2v.shape, order = 'F')]

	sel1 = (I>1).nonzero()
	II = concatenate((II,V[sel1]))
	JJ = concatenate((JJ,ij2v[np.ravel_multi_index((ones(sel1.shape,J[sel1])), dims = ij2v.shape, order = 'F')]))

	sel2 = (J>1).nonzero()
	II = concatenate((II,V[sel2]))
	JJ = concatenate((JJ,ij2v[np.ravel_multi_index((I[sel2],ones(sel2.shape)),dims = ij2v.shape, order = 'F')]))

	sel3 = nonzero((J>1) and (I>1))
	II = concatenate((II,V[sel3]))
	JJ = concatenate((JJ,ij2v[np.ravel_multi_index((ones(sel3.shape),ones(sel3.shape)),dims = ij2v.shape, order = 'F')]))

	bhat2d2 = csr_matrix((ones(II.shape), (II,JJ)),shape=(len(V),len(V2))).toarray()

	Bhat = sol.Bhat

	u,s,vh = linalg.svd(Bhat[2:,2:])
	v = v.T

	U = u[:,0:2]
	V = s[0:2,0:2] * v[:,0:2]
	R = Bhat[np.ravel_multi_index((I2[0:nspecial],J2[0:nspecial]), dims = Bhat.size, order = 'F')]

	Uchange = ones(U.shape)
	Uchange[0:2,0:2] = zeros(3,3)
	(iu, ju) = Uchange.nonzero()
	indu = np.ravel_multi_index((iu,ju), dims = U.shape, order = 'F')

	Vchange = ones(V.shape)
	(iv, jv) = Vchange.nonzero()
	indv = np.ravel_multi_index((iu,ju), dims = V.shape, order = 'F')

	nzr = len(R)
	nzu = Uchange.sum()
	nzv = Vchange.sum()

	indzr = range(1, nzr+1)
	indzu = range(nzr+1, nzr+nzu+1)
	indzv = range(nzr+nzu+1, nzr+nzu+nzv+1)

	param.U = U
	param.V = V
	param.R = R
	param.indu = indu
	param.indv = indv
	param.indzr = indzr
	param.indzu = indzu
	param. indzv = indzv
	parma.nzr = nzr
	param.mm = mm
	param.nn = nn
	mdata.d2vm = d2vm
	mdata.bhat2d2 = bhat2d2
	mdata.I22 = I22
	mdata.J22 = J22

	debug = 0

	res0, jac0 = calcresandjack(mdata,param)


	for kkk in range(0,5):

		res, jac = calcresandjack(mdata,param)

		dz = -(linalg.solve(a,b))

		param_new = updatexy(param, dz)
		res2, jac2 = calcresandjack(mdata,param_new)
		aa = concatenate((np.linalg.norm(res),np.linalg.norm(res+jac*dz), np.linalg.norm(res2)),1)
		bb = aa
		bb = bb - bb[1]
		bb = bb/bb[0]
		cc = np.linalg.norm(jac*dz)/np.linalg.norm(res)

		if np.linalg.norm(res) < np.linalg.norm(res2):

			if cc > 0.0001:

				kkkk = 1
				while (kkkk<50) and (np.linalg.norm(res)<np.linalg.norm(res2)):
					dz = dz/2
					param_new = updatexy(param,dz)
					res2, jac2 = calcresandjack(mdata, param_new)
					kkkk = kkkk+1


		if debug:
			aa = concatenate((np.linalg.norm(res), np.linalg.norm(res+jac*dz), np.linalg.norm(res2)),1)
			bb = aa
			bb = bb-bb[1]
			bb = bb/bb[0]
			cc = np.linalg.norm(jac*dz)/np.linalg.norm(res)

			print aa, bb, cc

		if np.linalg.norm(res2) < np.linalg.norm(res):
			param = param_new
		else:
			[]


	param_opt = param

	BB = zeros((mm,nn))
	BB[1:,1:] = (param_opt.U).conj().T * param_opt.V
	matsize = Bhat.shape
	rowSub = I2[0:nspecial]
	colSub = J2[0:nspecial]

	BB[np.ravel_multi_index((rowSub, colSub), dims = matsize, order = 'F')] = param_opt.R
	d2calc = zeros((d.shape))
	d2calc_shuffle = linalg.inv(dl) * BB * linalg.inv((dr.conj().T))

	sol.Bhat = BB

	return sol, res0, res, d2calc










