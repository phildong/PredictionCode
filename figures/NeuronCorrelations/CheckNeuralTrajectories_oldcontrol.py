
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
#Goal is to plot neural trajectories projected into first three PCs





for typ_cond in ['AML310_transition']: #, 'AML310_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # Hard coding in folder for old dataset:
    folder = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/'
    dataLog = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/Special_transition_datasets.txt'

    # data parameters
    dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
            'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5,  # gauss window for red and green channel
            'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
            'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
            }
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate = folder, dataPars = dataPars)
    keyList = np.sort(list(dataSets.keys()))
    theDataset = '172438'
    theDataset = '134913' #2018 dataset #hard Coded in
    for key in [x for x in keyList if theDataset in x]:
        print(("Running "+key))
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_withNaN = dataSets[key]['Neurons']['I_smooth'] # use this to find the untracked neurons after transition
        neurons_ZScore = dataSets[key]['Neurons']['ActivityFull'] # Z scored neurons to use to look at calcium traces
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        # curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
        # For immobile- how is NaN neurons that are not hand tracked dealt with by the smooth_interp...
        # Still do the correlation with all (the interpolated ones too, but then replace with 0s)?

    im_start = 1305
    im_end = 2205
    im_start = len(time_contig)/2
    im_end = len(time_contig)-1

    # see what index in time (noncontig) is closest to time_contig value
    start_time= time_contig[im_start]
    start_index = np.abs(time - start_time).argmin()
    end_time = time_contig[im_end]
    end_index = np.abs(time-end_time).argmin()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Plot neural state space trajectories in first 3 PCs
    # also reduce dimensionality of the neural dynamics.
    nComp = 3  # pars['nCompPCA']
    pca = PCA(n_components = nComp)
    Neuro = np.copy(neurons[:, start_index: end_index]).T

    # make sure data is centered
    sclar = StandardScaler(copy = True, with_mean = True, with_std = False)
    zscore = StandardScaler(copy = True, with_mean = True, with_std = True)
    Neuro_mean_sub = sclar.fit_transform(Neuro)
    Neuro_z = zscore.fit_transform(Neuro)

    pcs = pca.fit_transform(Neuro_mean_sub)
    pcs_z = pca.fit_transform(Neuro_z)
    idn = ''
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(key + idn + '\n PCA (minimally processed)')
    for nplot in np.arange(6) + 1:
        ax = plt.subplot(2, 3, nplot, projection='3d')
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2])
        ax.view_init(np.random.randint(360), np.random.randint(360))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    import prediction.provenance as prov
    prov.stamp(plt.gca(), .55, .15, __file__)

    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(key + idn + '\n PCA (z-scored) ')
    for nplot in np.arange(6) + 1:
        ax = plt.subplot(2, 3, nplot, projection='3d')
        ax.plot(pcs_z[:, 0], pcs_z[:, 1], pcs_z[:, 2], color='orange')
        ax.view_init(np.random.randint(360), np.random.randint(360))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    prov.stamp(plt.gca(), .55, .15, __file__)

    plt.figure()
    plt.plot(time[start_index:end_index], pcs[:, 0])
    plt.plot(time[start_index:end_index], pcs[:, 1])
    plt.plot(time[start_index:end_index], pcs[:, 2])


#    #We are goig to do PCA on a lot of Nan'ed out or interpolated timepoints.


#    plt.imshow(neurons_withNaN[:, im_start:im_end])
    plt.show()


print("Done")
