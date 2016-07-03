import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mean = [0,0,0]
cov_real = [[  5.25487551e-07,   4.14541382e-07,  -3.58051678e-08],
            [  4.14541382e-07,   6.92369518e-07,  -1.27248024e-08],
            [ -3.58051678e-08,  -1.27248024e-08,   1.93029537e-07]]
cov_est  = [[  5.45300741e-07,   3.96288567e-07,  -4.28965355e-08],
            [  3.96288567e-07,   6.28310654e-07,  -1.70455265e-08],
            [ -4.28965355e-08,  -1.70455265e-08,   1.81178674e-07]]
# real_sigmatx = 
# [[  8.00451118e-06   5.73122689e-06  -2.11095623e-07]
#  [  5.73122689e-06   8.49236228e-06  -8.85695117e-08]
#  [ -2.11095623e-07  -8.85695117e-08   2.76062893e-06]]
# avg_est_sigmatx = 
# [[  7.51396137e-06   5.59510532e-06  -2.90879563e-07]
#  [  5.59510532e-06   8.98672031e-06  -2.33677376e-07]
#  [ -2.90879563e-07  -2.33677376e-07   2.58350295e-06]]
# nsamples = 100
# x1, y1, z1 = np.random.multivariate_normal(mean, cov_real, nsamples).T
# x2, y2, z2 = np.random.multivariate_normal(mean, cov_est, nsamples).T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(nsamples):
#     ax.scatter(x1[i], y1[i], z1[i], c='b', marker='o')
#     ax.scatter(x2[i], y2[i], z2[i], c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show(False)

# eigen_val_1, eigen_vec_1 = np.linalg.eig(cov_real)
# print "Real ", "eigen_val ", eigen_val_1, " eigen_vec ",eigen_vec_1
# eigen_val_2, eigen_vec_2 = np.linalg.eig(cov_est)
# print "Est ", "eigen_val ", eigen_val_2, " eigen_vec ", eigen_vec_2

from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

points_real = np.random.multivariate_normal(mean = (0,0), cov = [[cov_real[1][1],cov_real[1][2]],[cov_real[2][1],cov_real[2][2]]], size =200)
x_real,y_real = points_real.T
plt.plot(x_real,y_real,'yo')
plot_point_cov(points_real,nstd=1,alpha=0.7,color='green')

points_est = np.random.multivariate_normal(mean = (0,0), cov = [[cov_est[1][1],cov_est[1][2]],[cov_est[2][1],cov_est[2][2]]], size =200)
x_est,y_est = points_est.T
plt.plot(x_est,y_est,'bo')

plot_point_cov(points_est,nstd=1,alpha=0.7,color='red')

plt.show(True)
