import numpy as np
import math
import transformation as tr
import COPE.SE3UncertaintyLib as SE3
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import IPython


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

def Jacxia_i(Ra_i, txhat):
    Jacxia = np.zeros((6,3))
    Jacxia[:3,:3] = np.eye(3)
    Jacxia[3:6,:3] = -SE3.Hat(np.dot(Ra_i,txhat))
    return Jacxia

ksamples =50
iters = 300
# True X
Rx = np.array([[ 0.97280147, -0.17360641,  0.15335616],
               [ 0.21505082,  0.92289897, -0.31939105],
               [-0.0860839 ,  0.34368345,  0.93513167]])
Rx = SE3.VecToRot(np.random.random_sample(3))
tx = np.array([0.341213,0.123214,-0.2])

sigmaA = 1e-8*np.diag((2 , 3, 4, 5, 5,4))
sigmaRa = sigmaA[3:,3:]
sigmata = sigmaA[:3,:3]
sigmaB = 1e-6*np.diag((2 , 5, 4, 0.09, 0.03,0.005))
sigmaRb = sigmaB[3:,3:]
sigmatb = sigmaB[:3,:3]

sigmaA = 1e-10*np.diag((1, 1, 1, 1, 1, 1))
sigmaRa = sigmaA[3:,3:]
sigmata = sigmaA[:3,:3]
diagonal = (0.0005, 0.003, 0.0005,0.025, 0.02, 0.05 )
diagonal = [e**2 for e in diagonal]
sigmaB = np.diag(diagonal)
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
        beta.append(SE3.RotToVec(np.dot(SE3.VecToRot(xi_beta),SE3.VecToRot(beta_true))))
        xi_alpha = np.random.multivariate_normal(np.zeros(3),sigmaRa)
        alpha_true = np.dot(Rx,beta_true)
        alpha.append(SE3.RotToVec(np.dot(SE3.VecToRot(xi_alpha),SE3.VecToRot(alpha_true))))

        ta_true = np.random.random_sample(3)
        xi_ta = np.random.multivariate_normal(np.zeros(3),sigmata)
        ta.append(ta_true + xi_ta)
        tb_true = np.dot(np.linalg.inv(Rx),ta_true-np.dot(np.eye(3)- SE3.VecToRot(alpha_true),tx))
        xi_tb= np.random.multivariate_normal(np.zeros(3),sigmatb)
        tb.append(tb_true + xi_tb)

    # Estimate Rx and its covariance
    # M matrix
    M = np.zeros(shape=(3,3))
    for i in range(ksamples):
        M = M + np.asmatrix(beta[i].reshape((3,1)))*np.asmatrix(alpha[i].reshape((3,1))).T
    eig_val,eig_vec = np.linalg.eig(M.T * M)
    # RotX
    Rxhat = np.asarray(eig_vec*np.diag(np.sqrt(1.0/eig_val))*np.linalg.inv(eig_vec)*M.T)
    # Compute CovRotX
    U = np.zeros(shape = (3,3))
    WZW = np.zeros(shape = (3,3))
    for i in range(ksamples):
        sigmaV_i = np.zeros((6,6))
        sigmaV_i[:3,:3] = np.dot(np.dot(SE3.VecToJacInv(beta[i]),sigmaRb),np.transpose(SE3.VecToJacInv(beta[i]))) 
        sigmaV_i[3:6,3:6] = np.dot(np.dot(SE3.VecToJacInv(alpha[i]),sigmaRa),np.transpose(SE3.VecToJacInv(alpha[i])))
        inv_sigmaV_i = np.linalg.inv(sigmaV_i)
        U += np.dot(np.dot(np.transpose(JacRx_i(Rxhat,beta[i])),inv_sigmaV_i),JacRx_i(Rxhat,beta[i]))
        W_i = np.dot(np.dot(np.transpose(JacRx_i(Rxhat,beta[i])),inv_sigmaV_i),Jacbeta_i(Rxhat))
        Z_i = np.dot(np.dot(np.transpose(Jacbeta_i(Rxhat)),inv_sigmaV_i),Jacbeta_i(Rxhat))
        inv_Z_i = np.linalg.inv(Z_i)
        WZW += np.dot(np.dot(W_i,inv_Z_i),np.transpose(W_i))
        # IPython.embed()
    sigmaRx = np.linalg.inv(U-WZW)
    sigmaRx_list.append(sigmaRx)
    xi_Rx_list.append(np.asarray(SE3.RotToVec(np.dot(Rxhat,np.linalg.inv(Rx)))).reshape(3))
    # IPython.embed()
    # Estimate tx and its covariance
    # C matrix
    C = np.eye(3)-SE3.VecToRot(alpha[0])
    for i in range(1,ksamples):
        C = np.vstack((C,np.eye(3)-SE3.VecToRot(alpha[i])))
    # g matrix
    g = ta[0] - np.dot(Rxhat,tb[0])
    for i in range(1,ksamples):
        g = np.vstack((g, ta[i] - np.dot(Rxhat,tb[i])))
    g = g.reshape(3*ksamples,1)
    # txhat
    txhat = np.dot(np.linalg.pinv(C),g).reshape(3)
    # Compute Cov of tx
    U = np.zeros(shape = (3,3))
    WZW = np.zeros(shape = (3,3))
    for i in range(ksamples):
        sigmaV_i = np.zeros((6,6))
        Rxhattbi = np.dot(Rxhat,tb[i])
        sigmaq_i = sigmata + np.dot(np.dot(Rxhat,sigmatb),np.transpose(Rxhat)) + np.dot(np.dot(SE3.Hat(Rxhattbi),sigmaRx),np.transpose(SE3.Hat(Rxhattbi)))
        sigmaV_i[:3,:3] = sigmaRa
        sigmaV_i[3:6,3:6] = sigmaq_i
        inv_sigmaV_i = np.linalg.inv(sigmaV_i)
        Ra_i = SE3.VecToRot(alpha[i])
        U += np.dot(np.dot(np.transpose(Jactx_i(Ra_i)),inv_sigmaV_i),Jactx_i(Ra_i))
        W_i = np.dot(np.dot(np.transpose(Jactx_i(Ra_i)),inv_sigmaV_i),Jacxia_i(Ra_i, txhat))
        Z_i = np.dot(np.dot(np.transpose(Jacxia_i(Ra_i, txhat)),inv_sigmaV_i),Jacxia_i(Ra_i, txhat))
        inv_Z_i = np.linalg.inv(Z_i)
        WZW += np.dot(np.dot(W_i,inv_Z_i),np.transpose(W_i))
    sigmatx = np.linalg.inv(U- WZW)
    sigmatx_list.append(sigmatx)
    xi_tx_list.append((txhat-tx).reshape(3))
    # IPython.embed()
        
real_sigmaRx = np.cov(np.transpose(xi_Rx_list))
avg_est_sigmaRx = np.average(sigmaRx_list,axis = 0)
print "Rotation", Rxhat
print "real_sigmaRx = \n", real_sigmaRx
print "avg_est_sigmaRx = \n", avg_est_sigmaRx

real_sigmatx = np.cov(np.transpose(xi_tx_list))
avg_est_sigmatx = np.average(sigmatx_list,axis = 0)
print "translation", txhat
print "real_sigmatx = \n", real_sigmatx
print "avg_est_sigmatx = \n", avg_est_sigmatx

cov_real = real_sigmaRx
cov_est =  avg_est_sigmaRx

# Compare y z
nstd=1
alpha = 0.5
mean = (0,0)
cov = cov_real
# cov = [[cov [0][0],cov [0][1]],[cov [1][0],cov [1][1]]]
cov = [[cov [1][1],cov [1][2]],[cov [2][1],cov [2][2]]]
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
# cov = [[cov [0][0],cov [0][1]],[cov [1][0],cov [1][1]]]
cov = [[cov [1][1],cov [1][2]],[cov [2][1],cov [2][2]]]
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
