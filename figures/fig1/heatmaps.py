"""
Created Wed 8 January 2020
Gaol here is to generate useful output to understand how ro why changes to preprocessing
are affecting the performance of the different model predictions.
by Andrew Leifer
leifer@princeton.edu
"""

################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np
import numpy.matlib

from prediction import user_tracker
import prediction.data_handler as dh

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats

import pandas as pd

import matplotlib.backends.backend_pdf
from matplotlib import ticker

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'x-large',
            'axes.titlesize': 'x-large',
            'xtick.labelsize': 'x-large',
            'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

codePath = userTracker.codePath()
outputFolder = os.path.join(codePath,'figures/output')

data = {}
for typ_cond in ['AKS297.51_moving', 'AML32_moving', 'AML18_moving']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '%s/' % typ_cond)
        dataLog = os.path.join(path, '{0}/{0}_datasets.txt'.format(typ_cond))
        dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)

        keyList = np.sort(dataSets.keys())
        data[typ_cond] = {}
        data[typ_cond]['dsets'] = keyList
        data[typ_cond]['input'] = dataSets


for key in ['AKS297.51_moving', 'AML32_moving',  'AML18_moving']:
    dset = data[key]['input']
    # For each recording
    for idn in dset.keys():
        dset = data[key]['input'][idn]

        #Get the relevant heatmaps
        I_smooth = dset['Neurons']['I_smooth']
        I_smooth_interp_crop_noncontig_wnans = np.copy(I_smooth)
        I_smooth_interp_crop_noncontig_wnans[:] = np.nan

        valid_map = dset['Neurons']['I_valid_map']
        I_smooth_interp_crop_noncontig_wnans[:, valid_map] = dset['Neurons']['I_smooth_interp_crop_noncontig']
        G = np.copy(I_smooth)
        G[:] = np.nan

        G[:, valid_map] = dset['Neurons']['G_smooth_interp_crop_noncontig']
        Gmeansub = G - np.matlib.repmat(np.nanmean(G, axis=1), G.shape[1], 1).T

        time = dset['Neurons']['I_Time']

        # Cluster on Z-scored interpolated data to get indices
        activity = dset['Neurons']['I_smooth_interp_crop_noncontig']
        zactivity = stats.zscore(activity, axis=1)
        Z = linkage(zactivity)
        d = dendrogram(Z, no_plot=True)
        idx_clust = np.array(d['leaves'])

        #Write out a CSV file with mapping from neuron index to clustered neuron index
        csv_out = os.path.join(outputFolder,  idn + 'indices.csv')
        df = pd.DataFrame(idx_clust)
        df.to_csv(csv_out, index=True, header='clustered index')
        print('generated', csv_out)

        prcntile = 99
        fig = plt.figure(figsize=(18,12), constrained_layout=False)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, height_ratios=[2, .7, .7], width_ratios=[5])
        fig.suptitle('data[' + key + '][' + idn + ']')

        ax = fig.add_subplot(gs[0, :])
        num_Neurons = I_smooth_interp_crop_noncontig_wnans.shape[0]
        vmax = np.nanpercentile(np.abs(Gmeansub), prcntile)
        vmin = -vmax

        pos = ax.imshow(I_smooth_interp_crop_noncontig_wnans[idx_clust,:], aspect='auto',
                        interpolation='none', vmin=vmin, vmax=vmax,
                        extent=[time[0], time[-1], -.5, num_Neurons-.5], origin='lower')
        ax.set_ylim(-.5, num_Neurons+.5)
        ax.set_yticks(np.arange(0, num_Neurons, 25))
        ax.set_xticks(np.arange(0, time[-1], 60))
        ax.set_title('I_smooth_interp_crop_noncontig_wnans  (smooth,  interpolated, common noise rejected, w/ large NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        cb = fig.colorbar(pos, ax=ax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        ax.set_xlim(0, time[-1])
        if idn == 'BrainScanner20200130_110803':
            AVAR = 32
            AVAL = 15

            AVAR_ci = np.argwhere(idx_clust == AVAR)
            AVAL_ci = np.argwhere(idx_clust == AVAL)

            yt = ax.get_yticks()
            yt = np.append(yt, [AVAR_ci, AVAL_ci])
            ytl = yt.tolist()
            ytl[-2:-1] = ["AVAR", "AVAL"]
            ax.set_yticks(yt)
            ax.set_yticklabels(ytl)

        beh = dset['BehaviorFull']['CMSVelocity']
        time = dset['Neurons']['TimeFull']

        axbeh = fig.add_subplot(gs[1,:])
        axbeh.plot(time, beh, linewidth=1.5, color='k')
        fig.colorbar(pos, ax=axbeh)
        axbeh.axhline(linewidth=0.5, color='k')
        axbeh.set_xlim(ax.get_xlim())

        axbeh.set_title('Velocity')
        axbeh.set_xlabel('Time (s)')
        axbeh.set_ylabel(r'$v$ (mm s$^{-1}$)')

        curv = dset['BehaviorFull']['Curvature']
        axbeh = fig.add_subplot(gs[2,:])
        axbeh.plot(time, curv, linewidth=1.5, color='brown')
        fig.colorbar(pos, ax=axbeh)
        axbeh.axhline(linewidth=.5, color='k')
        axbeh.set_yticks([-2*np.pi, 0, 2*np.pi])
        axbeh.set_yticklabels([r'$-2\pi$', '0',  r'$2\pi$'])
        axbeh.set_xlim(ax.get_xlim())

        axbeh.set_title('Curvature')
        axbeh.set_xlabel('Time (s)')
        axbeh.set_ylabel('$\kappa$ \n (rad bodylength$^{-1}$)')


print("Beginning to save heat maps")


filename = os.path.join(outputFolder, "heatmaps_clustered.pdf")
pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved heatmaps to: %s" % filename)