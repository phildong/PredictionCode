
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap



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
    drug_application = 620
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

        # Only consider neurons that have timepoints present for at least 75% of the time during immobilization
        frac_nan_during_imm = np.true_divide(np.sum(np.isnan(neurons_withNaN[:, im_start:]), axis=1),
                                             neurons_withNaN[:, transition:].shape[1])
        valid_imm = np.argwhere(frac_nan_during_imm < 0.25)[:, 0]


        dset = dataSets[key]
        Iz = neurons_ZScore
        # Cluster on Z-scored interpolated data to get indices
        # Cluster only on the immobile portion; and only consider neurons prsent for both moving and immobile
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(dataSets[key]['Neurons']['Activity'][valid_imm,transition:])
        d = dendrogram(Z, no_plot=True)
        idx_clust = np.array(d['leaves'])

        drug_app_time = time_contig[drug_application]
        drug_app_index =  np.abs(time - drug_app_time).argmin()
        imm_start_time = time_contig[im_start]
        imm_start_index = np.abs(time - imm_start_time).argmin() #Index is for Noncontig
        end_time = time_contig[im_end]
        end_index = np.abs(time - end_time).argmin() # INdex is for noncontig


    neurons.flags.writeable = False
    print(hash(neurons.data))

    idx_clust = np.arange(idx_clust.size)
    AVA1 = AVAR = 72#36
    AVA2 = AVAL = 22 #126#23
    AVAR_ci = np.squeeze(np.argwhere(valid_imm[idx_clust] == AVAR))
    AVAL_ci = np.squeeze(np.argwhere(valid_imm[idx_clust] == AVAL))

    neurons[valid_imm[idx_clust]]

    ######################################################
    #  Now we are lookikng at correlation structure between moving and immobile
    #########################################################
    def get_pearsons_coefs(recording):
        cc = np.corrcoef(recording)
        indices = np.tril_indices(cc.shape[0],-1) # get the lower triangle excluding the diagonel
        return cc[indices], cc

    neurons_v = neurons[valid_imm[idx_clust]]
    rho_mov, cmat_mov = get_pearsons_coefs(neurons_v[:, :drug_app_index])
    rho_imm, cmat_imm = get_pearsons_coefs(neurons_v[:, imm_start_index:])


    def calc_pdf(x, low_lim, high_lim, nbins):
        bin_width = np.true_divide(high_lim - low_lim, nbins)
        counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        density = np.true_divide(counts, np.sum(counts) * bin_width)
        return density, bin_centers, bin_edges

    low_lim, high_lim, nbins = -1, 1, 30
    rho_mov_pdf, x, binedges = calc_pdf(rho_mov, low_lim, high_lim, nbins)
    rho_imm_pdf, x, binedges = calc_pdf(rho_imm, low_lim, high_lim, nbins)

    from scipy import stats
    t, p = stats.ttest_ind(rho_mov, rho_imm)
    _, pw = stats.wilcoxon(rho_mov, rho_imm)

    plt.figure(figsize=(4,4))
    plt.step(x, rho_mov_pdf,  where='mid', label='Moving', lw=2)
    plt.step(x, rho_imm_pdf,  where='mid', label='Immobile', lw=2)
    plt.axvline(np.mean(rho_mov))
    plt.axvline(np.mean(rho_imm))
    plt.title('wilcoxon p=%.5f' % pw)
    plt.legend()


plt.show()
print("Done")
