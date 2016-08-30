import numpy as np
import math
import transformation as tr
import COPE.SE3UncertaintyLib as SE3
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import IPython

ksamples = 50
iters =500


def JQ_i(Ra_i):
    JQ_i = np.zeros((9,3))
    JQ_i[:3,:3] = -SE3.Hat(Ra_i[:3,0])
    JQ_i[3:6,:3] = -SE3.Hat(Ra_i[:3,1])
    JQ_i[6:9,:3] = -SE3.Hat(Ra_i[:3,2])
    return JQ_i

x = np.array([0.341213,0.123214,0.7])
Rxhat= SE3.VecToRot(x)
tx = np.array([1,2,3])
sigmaRx = 1e-4*np.diag([4,2,3])

sigmaA = 1e-4*np.diag((0.0004 , 0.002, 1., 0.1, 0.5,0.5))
sigmaRa = sigmaA[3:,3:]
sigmata = sigmaA[:3,:3]
sigmaB = 1e-4*np.diag((0.0003 , 0.0002, 0.0001, 1, 1, 1))
sigmaRb = sigmaB[3:,3:]
sigmatb = sigmaB[:3,:3]
tb_ = []
q = []
Q = []
beta_true = np.random.random_sample(3)
ta_true = np.random.random_sample(3)
for i in range(iters):
    xi_beta = np.random.multivariate_normal(np.zeros(3),sigmaRb)
    beta = xi_beta + beta_true
    Q.append(np.dot(SE3.VecToRot(xi_beta),SE3.VecToRot(beta_true)))

print np.cov(np.transpose([ Qi.flatten() for Qi in Q]))
print np.dot(np.dot(JQ_i(SE3.VecToRot(beta_true)),sigmaRb), np.transpose(JQ_i(SE3.VecToRot(beta_true))))
