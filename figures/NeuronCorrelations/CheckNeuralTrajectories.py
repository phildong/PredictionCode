
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
    theDataset = '172438'
    #theDataset = '134913' #2018 dataset #hard Coded in
    theDataset = '193044'
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

    im_start = 950
    im_end = 2885
    #im_start = len(time_contig)/2
    #im_end = len(time_contig)-1

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
    pca = PCA(n_components=nComp)
    Neuro = np.copy(neurons[:, start_index: end_index]).T

    #Repeat on the derivatives a la Kato et al
    def take_deriv(neurons):
        from prediction.Classifier import rectified_derivative
        _, _, nderiv = rectified_derivative(neurons)
        return nderiv
    Neuro_dFdt = take_deriv(neurons[:, start_index: end_index]).T


    # make sure data is centered
    sclar = StandardScaler(copy=True, with_mean=True, with_std=False)
    zscore = StandardScaler(copy=True, with_mean=True, with_std=True)
    Neuro_mean_sub = sclar.fit_transform(Neuro)
    Neuro_z = zscore.fit_transform(Neuro)
    Neuro_dFdt_mean_sub = sclar.fit_transform(Neuro_dFdt)
    Neuro_dFdt_z = zscore.fit_transform(Neuro_dFdt)


    pcs = pca.fit_transform(Neuro_mean_sub)

    AVA1 = 36
    AVA2 = 23
    print("AVA 1 weights:", pca.components_[:,AVA1])
    print("AVA 2 weights:", pca.components_[:,AVA2])

    pcs_z = pca.fit_transform(Neuro_z)
    pcs_dFdt = pca.fit_transform(Neuro_dFdt_mean_sub)
    pcs_dFdt_z = pca.fit_transform(Neuro_dFdt_z)

    plot_trajectories(pcs, key + '\n F PCA (minimally processed)')
    plot_trajectories(pcs_z, key + '\n F PCA (z-scored)')
    plot_trajectories(pcs_dFdt, key + '\n dF/dt PCA (minimally processed)', color='orange')
    plot_trajectories(pcs_dFdt_z, key + '\n dF/dt PCA (z-scored)', color='orange')

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time[start_index:end_index], pcs[:, 0], label='PC0')
    plt.plot(time[start_index:end_index], 3+pcs[:, 1], label='PC1')
    plt.plot(time[start_index:end_index], 6+pcs[:, 2], label='PC2')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[start_index:end_index], neurons[AVA1, start_index:end_index], label='Neuron %d' % AVA1)
    plt.plot(time[start_index:end_index], neurons[AVA2, start_index:end_index], label='Neuron %d' % AVA2)
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time[start_index:end_index], pcs_dFdt[:, 0], label='PC0 dF/dT')
    plt.plot(time[start_index:end_index], pcs_dFdt[:, 1], label='PC1 dF/dT')
    plt.plot(time[start_index:end_index], pcs_dFdt[:, 2], label='PC2 dF/dT')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[start_index:end_index], Neuro_dFdt[:, AVA1].T, label='Neuron %d' % AVA1)
    plt.plot(time[start_index:end_index], Neuro_dFdt[:, AVA2].T, label='Neuron %d' % AVA2)
    plt.legend()


#    #We are goig to do PCA on a lot of Nan'ed out or interpolated timepoints.


#    plt.imshow(neurons_withNaN[:, im_start:im_end])

    print("Plotting.")
    plt.show()


print("Done")
