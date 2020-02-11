import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import special
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

import dataHandler as dh
import userTracker

def sphere_distribution(neurons, z_axis = 0, n_theta = 50, n_phi = 50):
    thetas = np.linspace(0, np.pi, n_theta+1)
    phis = np.linspace(0, 2*np.pi, n_phi+1)
    theta, phi = np.meshgrid(thetas, phis)
    cts = np.zeros(theta.shape)

    neurons_z, neurons_x, neurons_y = neurons[:,z_axis], neurons[:,(z_axis+1) % 3], neurons[:,(z_axis+2) % 3]
    neurons_r = np.sqrt(neurons_z**2 + neurons_x**2 + neurons_y**2)
    neurons_theta = np.arccos(neurons_z/neurons_r)
    neurons_phi = np.arctan2(neurons_y, neurons_x)

    theta_idxs = np.floor(neurons_theta*n_theta/np.pi)
    phi_idxs = np.floor(neurons_phi*n_phi/(2*np.pi))

    for i in range(theta_idxs.size):
        cts[int(phi_idxs[i])][int(theta_idxs[i])] += 1
    
    return theta, phi, cts

def plot_sphere_heatmap(ax, thetas, phis, vals):
    x = np.sin(thetas)*np.cos(phis)
    y = np.sin(thetas)*np.sin(phis)
    z = np.cos(thetas)

    ax.plot_surface(x,y,z, rstride=1, cstride=1, facecolors=cm.Greys(vals/vals.max()), alpha=0.5, linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def trajectory_moments(neurons, z_axis = 0, num_moments = 15):
    theta, phi, cts = sphere_distribution(neurons, z_axis=z_axis)
    dtheta = np.pi/(theta.shape[0])
    dphi = 2*np.pi/(theta.shape[1])

    moment_vals = np.zeros(num_moments)
    for l in range(num_moments):
        val = 0
        for m in range(-l, l+1):
            yml = special.sph_harm(m, l, phi, theta)
            overlap = np.abs(np.sum(np.multiply(np.multiply(cts, yml), np.sin(theta)))*dtheta*dphi)
            normalize = np.sqrt(np.sum(np.multiply(np.multiply(cts, cts), np.sin(theta)))*dtheta*dphi)
            val += (overlap/normalize)**2
        moment_vals[l] = val

    exp_moment = np.dot(np.arange(num_moments), moment_vals)

    return moment_vals, exp_moment

def plot_moments(neurons_reduced, title):
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=(1, 1, 1, 1))
    ax_trajectories = []
    for i in range(3):
        ax_trajectories.append(fig.add_subplot(gs[i,0], projection='3d'))
    ax_spheres = []
    for i in range(3):
        ax_spheres.append(fig.add_subplot(gs[i,1], projection='3d'))
    ax_moments = []
    for i in range(3):
        ax_moments.append(fig.add_subplot(gs[i,2:]))

    score = 0
    for i in range(3):
        ax_trajectories[i].plot(neurons_reduced[:,(i+1) % 3], neurons_reduced[:,(i+2) % 3], neurons_reduced[:,i], c='k', alpha=0.5)
        labels = ['PCA1', 'PCA2', 'PCA3']
        ax_trajectories[i].set_zlabel(labels[i])
        ax_trajectories[i].set_xlabel(labels[(i+1) % 3])
        ax_trajectories[i].set_ylabel(labels[(i+2) % 3])
        ax_trajectories[i].set_xticks([])
        ax_trajectories[i].set_yticks([])
        ax_trajectories[i].set_zticks([])

        thetas, phis, cts = sphere_distribution(neurons_reduced, i)
        plot_sphere_heatmap(ax_spheres[i], thetas, phis, cts)

        moments, exp_moment = trajectory_moments(neurons_reduced, i)
        ax_moments[i].plot(moments)
        ax_moments[i].set_ylim(0, 0.6)
        ax_moments[i].set_xlabel(r'$\ell$')
        ax_moments[i].text(0.5, 0.5, r'$\langle \ell \rangle = %0.2f$' % exp_moment)
        score += exp_moment/3

    fig.suptitle(title + " Moment Distribution", fontsize=14)

    import prediction.provenance as prov
    prov.stamp(ax_moments[0],.5,1)

    return fig, score

def test_data1():
    t = np.linspace(0,100,2000)
    x = np.cos(t) + 0.1*np.cos(3.2*t)
    y = np.sin(t) + 0.1*np.sin(2.5*t)
    z = 0.5*np.cos(2*t) + 0.05*np.cos(7.8*t)

    return np.stack((z, x, y)).T

def test_data2():
    z = np.random.uniform(-1, 1, 2000)
    phis = np.random.uniform(0, 2*np.pi, 2000)
    x = np.sqrt(1-z*z)*np.cos(phis)
    y = np.sqrt(1-z*z)*np.sin(phis)

    return np.stack((z, x, y)).T

def main():
    plots = []
    plots.append((plot_moments(test_data1(), title = 'Structured Example'), -1))
    plots.append((plot_moments(test_data2(), title = 'Spherical Example'), -1))

    typ_conds = ['AML32_moving', 'AML32_immobilized', 'AML70_moving', 'AML70_chip', 'AML18_moving', 'AML18_immobilized']
    
    for typ_cond in typ_conds:
        path = userTracker.dataPath()
        folder = os.path.join(path, '%s/' % typ_cond)
        dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

        # data parameters
        dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
                'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
                'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
                'windowGCamp': 5,  # gauss window for red and green channel
                'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
                'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
                }
        dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
        keyList = np.sort(dataSets.keys())

        for key in keyList:
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
            
            pca = PCA(n_components=3)
            pca.fit(neurons.T)

            neurons_reduced = pca.transform(neurons.T)

            plots.append((plot_moments(neurons_reduced, title = typ_cond + " " + key), np.mean(velocity*velocity), np.std(curvature), typ_cond))

    
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "moment_distribution.pdf"))

    plots.sort(key=lambda x: x[0][1])
    for p in plots:
        pdf.savefig(p[0][0])

    colors = {k: v for k, v in zip(typ_conds, ['red', 'red', 'orange', 'orange', 'blue', 'blue'])}
    markers = {k: v for k, v in zip(typ_conds, ['o', '+', 'o', '^', 'o', '+'])}

    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    for typ_cond in typ_conds:
        ax.scatter(
            [x[0][1] for x in plots if x[1] >= 0 and x[3] == typ_cond], 
            [x[1] for x in plots if x[1] >= 0 and x[3] == typ_cond], 
            label=typ_cond, color = colors[typ_cond], marker=markers[typ_cond])
    ax.set_xlabel(r'Structure Metric $\langle\langle\ell\rangle\rangle$', fontsize=16)
    ax.set_ylabel('RMS Velocity', fontsize=16)
    ax.set_title('Velocity vs. Neural Structure')
    ax.set_xlim(1.3, 4)
    ax.set_ylim(-0.2, 3.5)
    ax.legend(fontsize=14)
    fig.savefig('velocity_vs_structure_all.png')

    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    for typ_cond in [typ_conds[i] for i in [0,2,3]]:
        ax.scatter(
            [x[0][1] for x in plots if x[1] >= 0 and x[3] == typ_cond], 
            [x[1] for x in plots if x[1] >= 0 and x[3] == typ_cond], 
            label=typ_cond, color = colors[typ_cond], marker=markers[typ_cond])
    ax.set_xlabel(r'Structure Metric $\langle\langle\ell\rangle\rangle$', fontsize=16)
    ax.set_ylabel('RMS Velocity', fontsize=16)
    ax.set_title('Velocity vs. Neural Structure')
    ax.set_xlim(1.3, 4)
    ax.set_ylim(-0.2, 3.5)
    ax.legend(fontsize=14)
    fig.savefig('velocity_vs_structure_moving.png')

if __name__ == '__main__':
    main()

