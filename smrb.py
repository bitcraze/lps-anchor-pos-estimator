import multipol
import numpy as np
import scipy.linalg
import tm_bundle_rank, tm_ransac_more_rows, tm_ransac_more_cols, toa_normalize, toa_3D_bundle
import toa_3D_bundle, polynomials2matrix


def system_misstoa_ransac_bundle(*argsin):
    
	dobundle = 1
	if len(argin) < 2:
		sys.ransac_threshold = 1
		sys.ransac_k = 70
		sys.ransac_threshold2 = 1
		sys.ransac_k2 = 20
		sys.min_inliers2 = 8

    d = argsin[0]
    sys = argsin[1]

	for kk in range(1, 2):

		sol, manrin = tm_ransac5rows(d,sys)

		if dobundle:
			sol, res0, res, d2calc = tm_bundle_rank(sol,d) 

		sol = tm_ransac_more_rows(d,sol,sys) 

		if dobundle:
			sol, res0, res, d2calc = tm_bundle_rank(sol,d) 

		sol = tm_ransac_more_cols(d,sol,sys) 

    	if dobundle:
        	sol, res0, res, d2calc = tm_bundle_rank(sol,d) 
    
    	sol = tm_ransac_more_rows(d,sol,sys) 

    	if dobundle:
        	sol, res0, res, d2calc = tm_bundle_rank(sol,d) 
   
   		sol = tm_ransac_more_cols(d,sol,sys) 

    	if dobundle:
        	sol, res0, res, d2calc = tm_bundle_rank(sol,d) 

    Bhat = sol.Bhat
    D = 3
    u, s, v = svd(Bhat[2:,2:])
    xr = u[:,0:D-1]
    yr = (s[0:D-1,0:D-1])*(v[:,0:D-1])
    auxvar1 = zeros((D,1))
    xtp = concatenate((auxvar1 xr))
    yt = concatenate((auxvar1 yr))
    xt = xtp / (-2) # maybe np.divide(xtp,-2)
    Bhatcol1 = Bhat[:,0]

    nr_of_unknowns = (D*(D+1)) / 2 + D +1
    E = eye(nr_of_unknowns)

    for i in range(1,nr_of_unknowns+1):
    	xv[i] = multipol(1, zeros(nr_of_unknowns, 1))

    one = multipol(1, zeros(nrofunknowns,1)) 
    zero = multipol(0,zeros(nrofunknowns,1))

    if D == 3:
        bv = np.ndarray([xv[6] xv[7] xv[9]])
        bv = bv.conj().T
    elif D == 2:
        bv = np.ndarray([xv[3] xv[5]])
        bv = bv.conj().T

    for i = 1:xt.shape[1]:
        eqs[i-1] = (-2 * (xt[:,i]).conj().T * bv + (xt[:,i]).conj().T * Cv * xt[:,i]) - Bhatcol1[i]

    eqs_linear = eqs
    cfm_linear, mons_linear = polynomials2matrix(eqs_linear) 
    cfm_linear = np.asarray(cfm_linear)
    cfm_linear = cfm_linear / tile(sqrt(sum(cfm_linear**2,2)),1, cfm_linear.shape[1])
    cfm_linear0 = cfm_linear
    AA = cfm_linear0[:, 1:-2]
    bb = cfm_linear0[:,-1]
    zz0 = -linalg.pinv(AA)*bb

    # H = evaluate(Cv, np.ndarray([zz0],[0]))
    # b = evaluate(bv,np.ndarray([zz0];[0]))

    if min(linalg.eig(H)):
        L = linalg.cholesky(linalg.inv(H)).T
    else:
        mins = min(linalg.eig(H))
        H = H + (-mins + 0.1) * eye(3)
        L = linalg.cholesky(linalg.inv(H)).T

    r00 = linalg.inv(L.conj().T) * xt
    s00 = L * (yt + tile(b, 1, Bhat.shape[1]))

    r0 = zeros(3,d.size[0])
    s0 = zeros(3,d.size[1])

    r0[:, sol.rows] = r00
    s0[:, sol.cols] = s00

    toa_calc_d_from_xy(r0, s0) 

    inliers = sol.inlmatrix==1

    r1, s1, res, jec = toa_3D_bundle(d, r0, s0, inliers) 
    r, s = toa_normalize(r1, s1) ##########

    return r, s, inliers











print __name__

