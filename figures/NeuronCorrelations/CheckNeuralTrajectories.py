
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
#Goal is to plot neural trajectories projected into first three PCs


def plot_trajectories(pcs, title='Neural State Space Trajectories', color='blue'):
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    for nplot in np.arange(6) + 1:
        ax = plt.subplot(2, 3, nplot, projection='3d')
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2])
        ax.view_init(np.random.randint(360), np.random.randint(360))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    import prediction.provenance as prov
    prov.stamp(plt.gca(), .55, .15, __file__)
    return


for typ_cond in ['AKS297.51_transition']: #, 'AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # Hard coding in folder for old dataset:
    #folder = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/'
    #dataLog = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/Special_transition_datasets.txt'

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
    theDataset = '193044'
    transition = im_start = 950
    im_end = 2885


    for key in filter(lambda x: theDataset in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_withNaN = dataSets[key]['Neurons']['I_smooth'] # use this to find the untracked neurons after transition
        neurons_ZScore = dataSets[key]['Neurons']['ActivityFull'] # Z scored neurons to use to look at calcium traces
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        # curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
        # For immobile- how is NaN neurons that are not hand tracked dealt with by the smooth_interp...
        # Still do the correlation with all (the interpolated ones too, but then replace with 0s)?

        dset = dataSets[key]
        Iz = neurons_ZScore
        # Cluster on Z-scored interpolated data to get indices
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(dataSets[key]['Neurons']['Activity'])
        d = dendrogram(Z[:,0:transition], no_plot=True)
        idx_clust = np.array(d['leaves'])

        imm_start_time = time_contig[im_start]
        imm_start_index = np.abs(time - imm_start_time).argmin()
        end_time = time_contig[im_end]
        end_index = np.abs(time - end_time).argmin()


    #### Plot heatmap and behavior for whoel recording
    fig = plt.figure(figsize=(18, 18), constrained_layout=False)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(ncols=1, nrows=5, figure=fig, height_ratios=[2, .7, .7, .7, .7], width_ratios=[5])
    ax = fig.add_subplot(gs[0, :])

    prcntile = 99.7
    num_Neurons=neurons_withNaN.shape[0]
    vmin = np.nanpercentile(neurons_withNaN, 0.1)
    vmax = np.nanpercentile(neurons_withNaN.flatten(), prcntile)
    pos = ax.imshow(neurons_withNaN[idx_clust, :], aspect='auto',
                    interpolation='none', vmin=vmin, vmax=vmax,
                    extent=[time_contig[0], time_contig[-1], -.5, num_Neurons - .5], origin='lower')
    ax.set_ylim(-.5, num_Neurons + .5)
    ax.set_yticks(np.arange(0, num_Neurons, 25))
    ax.set_xticks(np.arange(0, time_contig[-1], 60))
    # ax.set_title('I_smooth_interp_crop_noncontig_wnans  (smooth,  interpolated, common noise rejected, w/ large NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_xlim(0, end_time)
    from matplotlib import ticker

    cb = fig.colorbar(pos, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    AVA1 = AVAL = 72#36
    AVA2 = AVAR = 8 #126#23
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

    axbeh = fig.add_subplot(gs[1, :], sharex=ax)
    axbeh.plot(time, beh, linewidth=1.5, color='k')
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=0.5, color='k')
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(ax.get_xlim())
    axbeh.set_ylabel('Velocity')


    curv = dset['BehaviorFull']['Eigenworm3']
    axbeh = fig.add_subplot(gs[2, :], sharex=ax)
    axbeh.plot(time, curv, linewidth=1.5, color='brown')
    axbeh.set_ylabel('Curvature')
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=.5, color='k')
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(ax.get_xlim())

    axava1 = fig.add_subplot(gs[3, :], sharex=ax)
    axava1.plot(time_contig, dataSets[key]['Neurons']['gRaw'][AVAL,:])
    axava1.plot(time_contig, neurons_withNaN[AVAL,:])
    axava1.set_xticks(np.arange(0, time_contig[-1], 60))
    axava1.set_ylabel('AVAL')
    fig.colorbar(pos, ax=axava1)
    axava1.set_xlim(ax.get_xlim())

    axava = fig.add_subplot(gs[4, :], sharex=ax)
    axava.plot(time_contig, dataSets[key]['Neurons']['gRaw'][AVAR,:])
    axava.plot(time_contig, neurons_withNaN[AVAR,:])
    axava.set_ylabel('AVAR')
    axava.set_xticks(np.arange(0, time_contig[-1], 60))
    fig.colorbar(pos, ax=axava)
    axava.set_xlim(ax.get_xlim())

    ### Next it will be important to show that the neurons before and after transitions
    # are likely the same
    before = np.arange(700, 800)
    after = np.arange(1100,1200)
    av_b = np.nanmean(dataSets[key]['Neurons']['gRaw'][:, before], axis=1)
    av_a = np.nanmean(dataSets[key]['Neurons']['gRaw'][:, after], axis=1)
    av_bprime = np.nanmean(dataSets[key]['Neurons']['gRaw'][:, before-400], axis=1)
    av_aprime = np.nanmean(dataSets[key]['Neurons']['gRaw'][:, after+400], axis=1)
    plt.figure()
    for k in np.arange(av_b.shape[0]):
        plt.plot([0, 1], [av_b[k], av_a[k]],'ko-')
        plt.plot([3, 4], [av_bprime[k], av_b[k]],'ko-')
        plt.plot([5, 6], [av_a[k], av_aprime[k]],'ko-')
    plt.text(0, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_b - av_a)))
    plt.text(3, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_bprime - av_b)))
    plt.text(5, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_a - av_aprime)))
    labels = ['(700 to 800)', '(1100 to 1200)', '(300 to 400)', '(700 to 800)', '(1100 to 1200)', '(1500 to 1600)' ]
    plt.xticks([0, 1, 3, 4, 5, 6], labels)
    plt.title('Change in Mean raw RFP Values across different time windows')
    plt.ylabel('F')
    plt.xlabel('Averaging Window (Volumes)')


    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Plot neural state space trajectories in first 3 PCs
    # also reduce dimensionality of the neural dynamics.
    nComp = 3  # pars['nCompPCA']
    pca = PCA(n_components=nComp)
    Neuro = np.copy(neurons[:, imm_start_index: end_index]).T

    #Repeat on the derivatives a la Kato et al
    def take_deriv(neurons):
        from prediction.Classifier import rectified_derivative
        _, _, nderiv = rectified_derivative(neurons)
        return nderiv
    Neuro_dFdt = take_deriv(neurons[:, imm_start_index: end_index]).T


    # make sure data is centered
    sclar = StandardScaler(copy=True, with_mean=True, with_std=False)
    zscore = StandardScaler(copy=True, with_mean=True, with_std=True)
    Neuro_mean_sub = sclar.fit_transform(Neuro)
    Neuro_z = zscore.fit_transform(Neuro)
    Neuro_dFdt_mean_sub = sclar.fit_transform(Neuro_dFdt)
    Neuro_dFdt_z = zscore.fit_transform(Neuro_dFdt)


    pcs = pca.fit_transform(Neuro_mean_sub)


    print("AVA 1 weights:", pca.components_[:,AVA1])
    print("AVA 2 weights:", pca.components_[:,AVA2])
    print("AVA 1 ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVA1))
    print("AVA 1 ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVA2))

    print("Shape of pca.components_:", pca.components_.shape)

    pcs_z = pca.fit_transform(Neuro_z)
    pcs_dFdt = pca.fit_transform(Neuro_dFdt_mean_sub)
    
    print("AVA 1 weights:", pca.components_[:,AVA1])
    print("AVA 2 weights:", pca.components_[:,AVA2])
    print("AVA 1 ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVA1))
    print("AVA 1 ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVA2))
    
    pcs_dFdt_z = pca.fit_transform(Neuro_dFdt_z)

    plot_trajectories(pcs, key + '\n F PCA (minimally processed)')
    plot_trajectories(pcs_z, key + '\n F PCA (z-scored)')
    plot_trajectories(pcs_dFdt, key + '\n dF/dt PCA (minimally processed)', color='orange')
    plot_trajectories(pcs_dFdt_z, key + '\n dF/dt PCA (z-scored)', color='orange')

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time[imm_start_index:end_index], pcs[:, 0], label='PC0')
    plt.plot(time[imm_start_index:end_index], 3 + pcs[:, 1], label='PC1')
    plt.plot(time[imm_start_index:end_index], 6 + pcs[:, 2], label='PC2')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[imm_start_index:end_index], neurons[AVA1, imm_start_index:end_index], label='Neuron %d' % AVA1)
    plt.plot(time[imm_start_index:end_index], neurons[AVA2, imm_start_index:end_index], label='Neuron %d' % AVA2)
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time[imm_start_index:end_index], pcs_dFdt[:, 0], label='PC0 dF/dT')
    plt.plot(time[imm_start_index:end_index], pcs_dFdt[:, 1], label='PC1 dF/dT')
    plt.plot(time[imm_start_index:end_index], pcs_dFdt[:, 2], label='PC2 dF/dT')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[imm_start_index:end_index], Neuro_dFdt[:, AVA1].T, label='Neuron %d' % AVA1)
    plt.plot(time[imm_start_index:end_index], Neuro_dFdt[:, AVA2].T, label='Neuron %d' % AVA2)
    plt.legend()


#    #We are goig to do PCA on a lot of Nan'ed out or interpolated timepoints.


#    plt.imshow(neurons_withNaN[:, im_start:im_end])

    print("Plotting.")
    plt.show()


print("Done")
