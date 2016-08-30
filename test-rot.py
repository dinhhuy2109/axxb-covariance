import numpy as np
import math
import transformation as tr
import COPE.SE3UncertaintyLib as SE3
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import IPython

def compare_sigma(sigma1,sigma2):
    e = np.sqrt(np.trace(np.dot(np.transpose(sigma1-sigma2),(sigma1-sigma2))))
    return e

def compute_log(R):
    phi = math.acos((np.trace(R)-1)/2) #compute angle
    log = (phi/(2*math.sin(phi))*(R-R.T)) #compute log
    return np.matrix([[log[2,1]],[log[0,2]],[log[1,0]]])

def JacRx_i(Rxhat, beta_i):
    JacRx = np.zeros((6,3))
    JacRx[:3,:3] = np.zeros((3,3))
    JacRx[3:6,:3] = -SE3.Hat(np.dot(Rxhat,beta_i))
    return JacRx

def Jacbeta_i(Rxhat):
    Jacbeta =  np.zeros((6,3))
    Jacbeta[:3,:3] = np.eye(3)
    Jacbeta[3:6,:3] = Rxhat
    return Jacbeta

def Jactx_i(Ra_i):
    Jactx = np.zeros((6,3))
    Jactx[:3,:3] = np.zeros((3,3))
    Q_i = Ra_i - np.eye(3)
    Jactx[3:6,:3] = Q_i
    return Jactx

def JacQ_i(Ra_i, txhat):
    JacQ = np.zeros((6,3))
    JacQ[:3,:3] = np.eye(3)
    Q_i = Ra_i - np.eye(3)
    JacQ[3:6,:3] = -SE3.Hat(np.dot(Q_i,txhat))
    return JacQ

ksamples = 50
iters =300
# True X
Rx = np.array([[-0.17556632,  0.869125,    0.46238318],
               [ 0.95274827  ,0.03173922 , 0.30209826],
               [ 0.24788547 , 0.49357306 ,-0.83362967]])
tx = np.array([0.341213,0.123214,-0.2])
# Rx = tr.random_rotation_matrix()[:3,:3]

sigmaA = 1e-4*np.diag((0 , 0., 0., 0.1, 0.5,0.5))
sigmaRa = sigmaA[3:,3:]
sigmata = sigmaA[:3,:3]
sigmaB = 1e-8*np.diag((0. , 0., 0., 1, 1, 1))
# diagonal = (0.0005, 0.003, 0.0005,0.025, 0.02, 0.008 )
# diagonal = [e**2 for e in diagonal]
# sigmaB = np.diag(diagonal)
print sigmaB
sigmaRb = sigmaB[3:,3:]
sigmatb = sigmaB[:3,:3]
print "real X:\n", Rx
print "real tx", tx
xi_Rx_list = []
sigmaRx_list = []
xi_tx_list = []
sigmatx_list = []
for n in range(iters):
    alpha = []
    beta = []
    ta = []
    tb = []
    # Generate data
    for i in range(ksamples):
        beta_true = np.random.random_sample(3)
        xi_beta = np.random.multivariate_normal(np.zeros(3),sigmaRb)
        # beta.append(xi_beta + beta_true)
        # beta.append(SE3.RotToVec(np.dot(SE3.VecToRot(xi_beta),SE3.VecToRot(beta_true))))# +  xi_beta)
        xi_alpha = np.random.multivariate_normal(np.zeros(3),sigmaRa)
        alpha_true = np.dot(Rx,beta_true)
        # alpha.append(xi_alpha + alpha_true)
        # alpha.append(SE3.RotToVec(np.dot(SE3.VecToRot(xi_alpha),SE3.VecToRot(alpha_true))))
    # Estimate Rx and its covariance
    # M matrix
    M = np.zeros(shape=(3,3))
    for i in range(ksamples):
        M = M + np.asmatrix(beta[i].reshape((3,1)))*np.asmatrix(alpha[i].reshape((3,1))).T
    eig_val,eig_vec = np.linalg.eig(M.T * M)
    # RotX
    Rxhat = np.asarray(eig_vec*np.diag(np.sqrt(1.0/eig_val))*np.linalg.inv(eig_vec)*M.T)
    # print "Rotation", Rxhat
    # Compute CovRotX
    U = np.zeros(shape = (3,3))
    WZW = np.zeros(shape = (3,3))
    for i in range(ksamples):
        sigmaV_i = np.zeros((6,6))
        sigmaV_i[:3,:3] = sigmaRb
        sigmaV_i[3:6,3:6] = sigmaRa
        # sigmaV_i[3:6,3:6] = np.dot(np.dot(SE3.VecToJacInv(x),sigma),np.transpose(SE3.VecToJacInv(x)))
        inv_sigmaV_i = np.linalg.inv(sigmaV_i)
        # print "inv_sigmaV_i\n", inv_sigmaV_i
        U += np.dot(np.dot(np.transpose(JacRx_i(Rxhat,beta[i])),inv_sigmaV_i),JacRx_i(Rxhat,beta[i]))
        W_i = np.dot(np.dot(np.transpose(JacRx_i(Rxhat,beta[i])),inv_sigmaV_i),Jacbeta_i(Rxhat))
        Z_i = np.dot(np.dot(np.transpose(Jacbeta_i(Rxhat)),inv_sigmaV_i),Jacbeta_i(Rxhat))
        # print "JacRx_i(Rxhat,beta[i])\n" , JacRx_i(Rxhat,beta[i])
        # print "np.transpose(JacRx_i(Rxhat,beta[i]))\n", np.transpose(JacRx_i(Rxhat,beta[i]))
        # print "Jacbeta_i(Rxhat)\n", Jacbeta_i(Rxhat)
        # print "W_i\n", W_i
        # print "Z_i\n",Z_i
        # IPython.embed()
        inv_Z_i = np.linalg.inv(Z_i)
        WZW += np.dot(np.dot(W_i,inv_Z_i),np.transpose(W_i))
        
    # print "U\n", U
    # print "WZW\n", WZW
    sigmaRx = np.linalg.inv(U-WZW)
    sigmaRx_list.append(sigmaRx)
    xi_Rx_list.append(np.asarray(compute_log(np.dot(Rxhat,np.linalg.inv(Rx)))).reshape(3))
    # IPython.embed()
real_sigmaRx = np.cov(np.transpose(xi_Rx_list))
avg_est_sigmaRx = np.average(sigmaRx_list,axis = 0)
print "Rotation", Rxhat
print "real_sigmaRx = \n", real_sigmaRx
print "avg_est_sigmaRx = \n", avg_est_sigmaRx
print "Error = ", compare_sigma(avg_est_sigmaRx,real_sigmaRx) 

# nsamples = 100
# a = [SE3.RotToVec(np.dot(SE3.VecToRot(xi_Rx),Rx)) for xi_Rx in xi_Rx_list]
# x1 = [aa[0] for aa in a]
# y1 = [aa[1] for aa in a]
# z1 = [aa[2] for aa in a]
# x2, y2, z2 = np.random.multivariate_normal(np.asarray(compute_log(Rx)).reshape(3), avg_est_sigmaRx, nsamples).T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(nsamples):
#     ax.scatter(x1[i], y1[i], z1[i], c='r', marker='o')
#     ax.scatter(x2[i], y2[i], z2[i], c='b', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.axis('equal')
# plt.show(False)






cov_real = real_sigmaRx
cov_est =  avg_est_sigmaRx

# Compare y z
nstd=1
alpha = 0.5
mean = (0,0)
cov = cov_real
cov = [[cov [0][0],cov [0][1]],[cov [1][0],cov [1][1]]]
# cov = [[cov [1][1],cov [1][2]],[cov [2][1],cov [2][2]]]
# cov = [[cov [0][0],cov [0][2]],[cov [2][0],cov [2][2]]]
points  = np.random.multivariate_normal(mean,cov, size =100)
x ,y  = points .T
plt.plot(x ,y ,'y.')
pos = mean

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

ax = plt.gca()
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
# Width and height are "full" widths, not radius
width, height = 2 * nstd * np.sqrt(vals)
ellip1 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='green',linewidth=2, fill=False,label = 'Monte Carlo result')
ax.add_artist(ellip1)

mean = (0,0)
cov = cov_est
cov = [[cov [0][0],cov [0][1]],[cov [1][0],cov [1][1]]]
# cov = [[cov [1][1],cov [1][2]],[cov [2][1],cov [2][2]]]
# cov = [[cov [0][0],cov [0][2]],[cov [2][0],cov [2][2]]]
points_est = np.random.multivariate_normal(mean, cov , size =100)
x_est,y_est = points_est.T
plt.plot(x_est,y_est,'b.')

pos=mean
ax = plt.gca()
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
# Width and height are "full" widths, not radius
width, height = 2 * nstd * np.sqrt(vals)
ellip2 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False,label ='Estimated covariance')
ax.add_artist(ellip2)

plt.xlabel(r'${\bf{\xi}}_{1}$',fontsize=25, labelpad=10)
plt.ylabel(r'${\bf{\xi}}_{2}$',fontsize=25, labelpad=5)
plt.legend(handles=[ellip1, ellip2])
ax.set(aspect='equal')
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 0.004))
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 0.004))
plt.show(False)
