"""
Created 23 Nov 2020
Generate all the heatmaps in one giant figure for the supplement
leifer@princeton.edu
"""

################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh






def main():

    codePath = userTracker.codePath()
    outputFolder = os.path.join(codePath,'figures/subpanels_revision/generatedFigs')
    maxTime=730

    data = {}
    for typ in ['AKS297.51', 'AML32', 'AML18']:
        for condition in ['moving', 'chip', 'immobilized']:  # ['moving', 'immobilized', 'chip']:
            path = userTracker.dataPath()
            folder = os.path.join(path, '{}_{}/'.format(typ, condition))
            dataLog = os.path.join(path, '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
            outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
            outLocData = os.path.join(path, 'Analysis/{}_{}.hdf5'.format(typ, condition))

            try:
                # load multiple datasets
                dataSets = dh.loadDictFromHDF(outLocData)
                keyList = np.sort(dataSets.keys())
                results = dh.loadDictFromHDF(outLoc)
                # store in dictionary by typ and condition
                key = '{}_{}'.format(typ, condition)
                data[key] = {}
                data[key]['dsets'] = keyList
                data[key]['input'] = dataSets
                data[key]['analysis'] = results
            except IOError:
                print typ, condition, 'not found.'
                pass

    print 'Done reading data.'

    import matplotlib.pyplot as plt




    ### Plot Heatmap for each recording





    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

    Nrecs=0
    lengths=[]
    for key in ['AKS297.51_moving', 'AML32_moving',  'AML18_moving']:
        dset = data[key]['input']
        # For each recording
        for idn in dset.keys():
            Nrecs = Nrecs + 1
            time = dset[idn]['Neurons']['I_Time']
            lengths.append(time[-1]-time[0])


    fig = plt.figure(figsize=(18, 12*Nrecs), constrained_layout=False)
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(ncols=1, nrows=3*Nrecs, figure=fig, height_ratios=[2, .7, .7]*Nrecs, width_ratios=[5])

    print("Plotting heatmaps.....")
    sp_indx = 0
    for key in ['AKS297.51_moving', 'AML32_moving',  'AML18_moving']:
        dset = data[key]['input']
        # For each recording
        for idn in dset.keys():

            dset = data[key]['input'][idn]


            #Get the relevant heatmaps
            I_smooth = dset['Neurons']['I_smooth']
            I_smooth_interp_crop_noncontig_wnans = np.copy(I_smooth)
            I_smooth_interp_crop_noncontig_wnans[:] = np.nan
            G = np.copy(I_smooth)
            G[:] = np.nan


            valid_map = dset['Neurons']['I_valid_map']
            I_smooth_interp_crop_noncontig_wnans[:, valid_map] = dset['Neurons']['I_smooth_interp_crop_noncontig']
            G[:, valid_map] = dset['Neurons']['G_smooth_interp_crop_noncontig']
            import numpy.matlib
            Gmeansub = G - np.matlib.repmat(np.nanmean(G, axis=1), G.shape[1], 1).T


            time = dset['Neurons']['I_Time']

            # Cluster on Z-scored interpolated data to get indices
            from scipy.cluster.hierarchy import linkage, dendrogram
            Z = linkage(dset['Neurons']['Activity'])
            d = dendrogram(Z, no_plot=True)
            idx_clust = np.array(d['leaves'])

            prcntile = 99.7
            prcntile = 99


            ax = fig.add_subplot(gs[sp_indx, :])
            sp_indx = sp_indx+1
            ax.set_title('data[' + key + '][' + idn + ']')

            num_Neurons = I_smooth_interp_crop_noncontig_wnans.shape[0]


            vmax = np.nanpercentile(np.abs(Gmeansub), prcntile)
            vmin = -vmax

            pos = ax.imshow(I_smooth_interp_crop_noncontig_wnans[idx_clust,:], aspect='auto',
                            interpolation='none', vmin=vmin, vmax=vmax,
                            extent=[time[0], time[-1], -.5, num_Neurons-.5], origin='lower')
            ax.set_ylim(-.5, num_Neurons+.5)
            ax.set_yticks(np.arange(0, num_Neurons, 25))
            ax.set_xticks(np.arange(0, time[-1], 60))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            from matplotlib import ticker
            cb = fig.colorbar(pos, ax=ax)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            cb.update_ticks()

            ax.set_xlim(0, maxTime)
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

            beh = dset['BehaviorFull']['AngleVelocity']
            time = dset['Neurons']['TimeFull']

            axbeh = fig.add_subplot(gs[sp_indx,:])
            sp_indx=sp_indx+1
            axbeh.plot(time, beh, linewidth=1.5, color='k')
            fig.colorbar(pos, ax=axbeh)
            axbeh.axhline(linewidth=0.5, color='k')
            axbeh.set_xlim(ax.get_xlim())

            #axbeh.set_title('Velocity')
            #axbeh.set_xlabel('Time (s)')
            #axbeh.set_ylabel('Body Bend Velocity (radians per second)')
            from prediction import provenance as prov
            #prov.stamp(plt.gca(), .9, .15, __file__)

            curv = dset['BehaviorFull']['Eigenworm3']
            axbeh = fig.add_subplot(gs[sp_indx,:])
            sp_indx=sp_indx+1
            axbeh.plot(time, curv, linewidth=1.5, color='brown')
            fig.colorbar(pos, ax=axbeh)
            axbeh.axhline(linewidth=.5, color='k')
            axbeh.set_xlim(ax.get_xlim())

            axbeh.set_title('Velocity')
            axbeh.set_xlabel('Time (s)')
            #axbeh.set_ylabel('Curvature (arb units)')
            from prediction import provenance as prov
            #prov.stamp(plt.gca(), .9, .15, __file__)

    plt.tight_layout()
    print("Beginning to save heat maps")
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, "heatmaps_clustered.pdf"))
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved heatmaps.")





if __name__ == '__main__':
    print("about to run main()")
    main()