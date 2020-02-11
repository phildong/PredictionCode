import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

import dataHandler as dh
import userTracker

def pca_map(neurons, velocity, curvature):
    pca = PCA(n_components=2)
    pca.fit(neurons.T)

    neurons_reduced = pca.transform(neurons.T)
    minx, maxx = np.min(neurons_reduced[:,0])-1, np.max(neurons_reduced[:,0])+1
    miny, maxy = np.min(neurons_reduced[:,1])-1, np.max(neurons_reduced[:,1])+1

    mesh_count_x = 50
    mesh_count_y = 50
    i = lambda x: int((x-minx)/(maxx-minx)*mesh_count_x)
    j = lambda y: int((y-miny)/(maxy-miny)*mesh_count_y)
    vels = [[[] for _ in range(mesh_count_x)] for _ in range(mesh_count_y)]
    curvs = [[[] for _ in range(mesh_count_x)] for _ in range(mesh_count_y)]
    for a in range(neurons_reduced.shape[0]):
        vels[j(neurons_reduced[a,1])][i(neurons_reduced[a,0])].append(velocity[a])
        curvs[j(neurons_reduced[a,1])][i(neurons_reduced[a,0])].append(curvature[a])
    
    vel_means = np.array([[np.mean(vels[jj][ii]) for ii in range(mesh_count_x)] for jj in range(mesh_count_y)])
    vel_stds = np.array([[np.std(vels[jj][ii]) for ii in range(mesh_count_x)] for jj in range(mesh_count_y)])
    
    curv_means = np.array([[np.mean(curvs[jj][ii]) for ii in range(mesh_count_x)] for jj in range(mesh_count_y)])
    curv_stds = np.array([[np.std(curvs[jj][ii]) for ii in range(mesh_count_x)] for jj in range(mesh_count_y)])

    return vel_means, vel_stds, curv_means, curv_stds, neurons_reduced

def mean_std_to_rgb(Z):
    X = np.real(Z)
    Y = np.imag(Z)

    min_mean, max_mean = np.min(X), np.max(X)
    hsv = np.zeros((X.shape[0], X.shape[1], 3), dtype='float')
    hsv[..., 0] = 2/3-(X-min_mean)/(max_mean-min_mean)
    hsv[..., 1] = np.clip(Y/np.abs(X), 0, 1)
    hsv[..., 2] = 1
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def main():
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "pca_maps.pdf"))

    for typ_cond in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML18_moving']:
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

            vel_means, vel_stds, curv_means, curv_stds, neurons_reduced = pca_map(neurons, velocity, curvature)
            fig, ax = plt.subplots(2,3,figsize=(15,11))
            ax[0,0].scatter(neurons_reduced[:,0], neurons_reduced[:,1], c='k', marker='o', alpha=0.1)
            ax[0,0].set_title("Neural State Space Trajectory")
            
            ax[1,0].scatter(neurons_reduced[:,0], neurons_reduced[:,1], c='k', marker='o', alpha=0.1)
            ax[1,0].set_title("Neural State Space Trajectory")

            sns.set_palette("RdBu")

            sns.heatmap(vel_means, ax = ax[0,1], mask=np.isnan(vel_means), vmin=-3.5, vmax=3.5)
            ax[0,1].set_title("Velocity")
            ax[0,1].invert_yaxis()
            
            sns.heatmap(vel_stds, ax = ax[0,2], mask=np.isnan(vel_stds), vmin=0, vmax=2)
            ax[0,2].set_title("Velocity Standard Deviations")
            ax[0,2].invert_yaxis()

            sns.heatmap(curv_means, ax = ax[1,1], mask=np.isnan(curv_means), vmin=-24, vmax=24)
            ax[1,1].set_title("Curvature")
            ax[1,1].invert_yaxis()
            
            sns.heatmap(curv_stds, ax = ax[1,2], mask=np.isnan(curv_stds), vmin=0, vmax=15)
            ax[1,2].set_title("Curvature Standard Deviations")
            ax[1,2].invert_yaxis()

            fig.suptitle("%s %s PCA Analysis" % (typ_cond, key), fontsize=14)

            import prediction.provenance as prov
            prov.stamp(ax[1,2],.55,-.15)

            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

if __name__ == '__main__':
    main()

