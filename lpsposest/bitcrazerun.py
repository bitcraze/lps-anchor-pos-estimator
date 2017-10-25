#
#    ||          ____  _ __
# +------+      / __ )(_) /_______________ _____  ___
# | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#
# LPS Anchor Position Estimator
#
# Copyright (C) 2016 Bitcraze AB
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,USA
#
import numpy as np
import scipy.io as sio
from smrb import system_misstoa_ransac_bundle
from toa_3D_bundle_with_smoother import toa_3D_bundle_with_smoother
from toa_calc_d_from_xy import toa_calc_d_from_xy
from toa_normalize import toa_normalize


data_file = sio.loadmat('testbitcrazerun.mat')

d = data_file['d']


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
