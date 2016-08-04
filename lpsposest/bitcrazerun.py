import numpy as np
import scipy.io as sio
from toa_3D_bundle_with_smoother import *
from smrb import system_misstoa_ransac_bundle
from toa_calc_d_from_xy import toa_calc_d_from_xy
from toa_normalize import toa_normalize



def bitcrazerun():
    class structtype():
        pass

    sys = structtype()

    sys.ransac_threshold = 2
    sys.ransac_k = 70
    sys.ransac_threshold2 = 2
    sys.ransac_k2 = 20
    sys.min_inliers2 = 8

    r, s, inliers = system_misstoa_ransac_bundle(d, sys)

    dcalc = toa_calc_d_from_xy(r, s)
    resm = dcalc - d
    inl2 = (abs(resm) < 1)

    mid = range(2, (d.shape[1]))

    sys.cc = (np.concatenate((mid - 1, mid, mid + 1))).conj().T
    sys.lambdacc = 1
    r2, s2, res, jac = toa_3D_bundle_with_smoother(d, r, s, inl2, sys)

    dcalc = toa_calc_d_from_xy(r2, s2)
    resm = dcalc - d

    inl = (abs(resm) < 0.3)

    mid = range(2, d.shape[1])
    sys.cc = (np.concatenate((mid - 1, mid, mid + 1))).conj().T
    sys.lambdacc = 3
    r3, s3, res, jac = toa_3D_bundle_with_smoother(d, r, s, inl, sys)

    anchors, flightpath = toa_normalize(r3, s3)

    A1 = np.concatenate((range(0, anchors.shape[1]), anchors))

    fileID = open('anchor_pos.yaml', 'w')
    fileID.write('n_anchors: %s\n', str(anchors.shape[1]))
    formatSpec = 'anchor%s_pos: [%s, %s, %s]\n'
    fileID.write(formatSpec, A1)
    fileID.close()
