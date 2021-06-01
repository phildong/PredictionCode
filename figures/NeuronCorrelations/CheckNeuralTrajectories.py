
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
#Goal is to plot neural trajectories projected into first three PCs

def plot_a_trajectory(ax, pc_traj, tscale, theta=0, phi=0, color='#1f77b4', gradient = True):
    ax.view_init(theta, phi)
    
    if gradient:
        lns = pc_traj.shape[0]-1
        #print(tscale)
        for i in range(lns):
            ax.plot(pc_traj[i:i+2, 0], pc_traj[i:i+2, 1], pc_traj[i:i+2, 2], color=plt.cm.gist_ncar(tscale[i]))
    else:
        ax.plot(pc_traj[:,0], pc_traj[:,1], pc_traj[:,2], color=color)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

def sync3d_limits(ax1, ax2):
    xmax, xmin = np.max([ax1.axes.get_xlim3d(), ax2.axes.get_xlim3d()]), np.min([ax1.axes.get_xlim3d(), ax2.axes.get_xlim3d()])
    ymax, ymin = np.max([ax1.axes.get_ylim3d(), ax2.axes.get_ylim3d()]), np.min([ax1.axes.get_ylim3d(), ax2.axes.get_ylim3d()])
    zmax, zmin = np.max([ax1.axes.get_zlim3d(), ax2.axes.get_zlim3d()]), np.min([ax1.axes.get_zlim3d(), ax2.axes.get_zlim3d()])
    ax1.axes.set_xlim3d(xmin, xmax)
    ax2.axes.set_xlim3d(xmin, xmax)
    ax1.axes.set_ylim3d(ymin, ymax)
    ax2.axes.set_ylim3d(ymin, ymax)
    ax1.axes.set_zlim3d(zmin, zmax)
    ax2.axes.set_zlim3d(zmin, zmax)




def plot_trajectories(pc_traj, time, drug_app_index, imm_start_index, end_index, title='Neural State Space Trajectories', color='#1f77b4', theta=None, phi =None):
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    row=2
    col=4
    #How do we choose the orientation of the 3d plot?
    if theta is None or phi is None:
        RAND = True #randomize theta or phi
    else:
        RAND = False # show the plot orientated around the specified value
        theta_spec, phi_spec = theta, phi
        jit = 1.5 #magnitude of jitter in degrees

    tscale = (time-time[0]+.0)/(time[-1]-time[0])
    for nplot in np.arange(col) + 1:
        if RAND: #Generate a random angle to view the 3D plot
            theta, phi = np.random.randint(360), np.random.randint(360)
        else: #Generate a random centered around the view
            theta, phi = theta_spec + jit*np.random.randn(), phi_spec + jit*np.random.randn()
        ax1 = plt.subplot(row, col, nplot, projection='3d', title='immobile (%d, %d)' % (theta, phi) )
        plot_a_trajectory(ax1, pc_traj[imm_start_index:end_index,:], tscale[imm_start_index:end_index], theta, phi, color)

        ax2 = plt.subplot(row, col, nplot+col, projection='3d', title='moving (%d, %d)' % (theta, phi))
        plot_a_trajectory(ax2, pc_traj[:drug_app_index,:], tscale[:drug_app_index], theta, phi, color)
        sync3d_limits(ax1, ax2)

    cfig = plt.figure()
    cax = plt.subplot(1, 1, 1)
    cax.set_axis_off()
    cax.imshow(np.vstack((tscale, tscale)), aspect=10, cmap=plt.cm.gist_ncar)
    cfig.savefig(os.path.join(userTracker.codePath(), 'figures/subpanels_revision/generatedFigs/')+'colorbar_CheckNeuralTrajectories.pdf')

    import prediction.provenance as prov
    #prov.stamp(plt.gca(), .55, .15, __file__)
    return


for typ_cond in ['AKS297.51_transition']: #, 'AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # Hard coding in folder for old dataset:
    #folder = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/'
    #dataLog = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/Special_transition_datasets.txt'

    # data parameters

    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
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
        G = dataSets[key]['Neurons']['G_smooth']
        import numpy.matlib
        Gmeansub = G - np.matlib.repmat(np.nanmean(G, axis=1), G.shape[1], 1).T
        velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']

        # Only consider neurons that have timepoints present for at least 75% of the time during immobilization
        frac_nan_during_imm = np.true_divide(np.sum(np.isnan(neurons_withNaN[:, im_start:]), axis=1),
                                             neurons_withNaN[:, transition:].shape[1])
        valid_imm = np.argwhere(frac_nan_during_imm < 0.25)[:, 0]
        valid_imm = valid_imm[valid_imm != 115] #Have to remove neuron 115 because found a tracking error

        dset = dataSets[key]
        # Cluster on Z-scored interpolated data to get indices
        # Cluster only on the immobile portion; and only consider neurons prsent for both moving and immobile
        from scipy.cluster.hierarchy import linkage, dendrogram
        rel_activity=dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig'][valid_imm,transition:]
        from scipy import stats
        rel_zactivity = stats.zscore(rel_activity, axis=1)
        Z = linkage(rel_zactivity)
        d = dendrogram(Z, no_plot=True)
        idx_clust = np.array(d['leaves'])

        drug_app_time = time_contig[drug_application]
        drug_app_index =  np.abs(time - drug_app_time).argmin()
        imm_start_time = time_contig[im_start]
        imm_start_index = np.abs(time - imm_start_time).argmin() #Index is for Noncontig
        end_time = time_contig[im_end]
        end_index = np.abs(time - end_time).argmin() # INdex is for noncontig





    #### Plot heatmap and behavior for whoel recording
    fig = plt.figure(figsize=(10, 12), constrained_layout=False)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(ncols=1, nrows=6, figure=fig, height_ratios=[2, .6, .6, .8, 1.1, 1.3], width_ratios=[5])
    ax = fig.add_subplot(gs[0, :])

    prcntile = 99.7
    num_Neurons=neurons_withNaN[valid_imm, :].shape[0]
    vmax = np.nanpercentile(np.abs(Gmeansub), prcntile)
    vmin = -vmax
    pos = ax.imshow(neurons_withNaN[valid_imm[idx_clust], :], aspect='auto',
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

    AVA1 = AVAR = 72#36
    AVA2 = AVAL = 22 #126#23
    AVAR_ci = np.squeeze(np.argwhere(valid_imm[idx_clust] == AVAR))
    AVAL_ci = np.squeeze(np.argwhere(valid_imm[idx_clust] == AVAL))

    yt = ax.get_yticks()
    yt = np.append(yt, [AVAR_ci, AVAL_ci])
    ytl = yt.tolist()
    ytl[-2:-1] = ["AVAR", "AVAL"]
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)


    curv_ylims=[-9, 9]
    vel_ylims=[-.2, .3]
    axbeh = fig.add_subplot(gs[1, :], sharex=ax)
    beh = dset['BehaviorFull']['CMSVelocity']
    axbeh.plot(dset['Neurons']['TimeFull'], beh, linewidth=1.5, color='k')
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=0.5, color='k')
    axbeh.axvline(drug_app_time)
    axbeh.axvline(imm_start_time)
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(0, end_time)
    axbeh.set_ylabel('Velocity')
    axbeh.set_ylim(vel_ylims)
    axbeh.axes.get_xaxis().set_visible(False)

    curv = dset['BehaviorFull']['Curvature']
    axbeh = fig.add_subplot(gs[2, :], sharex=ax)
    axbeh.plot(dset['Neurons']['TimeFull'], curv, linewidth=1.5, color='brown')
    axbeh.set_ylabel('Curvature')
    axbeh.set_ylim(curv_ylims)
    axbeh.set_yticks([-2 * np.pi, 0, 2 * np.pi])
    axbeh.set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'])
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=.5, color='k')
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(0, end_time)
    axbeh.axes.get_xaxis().set_visible(False)

    axava1 = fig.add_subplot(gs[3, :], sharex=ax)
    #axava1.plot(time_contig, dataSets[key]['Neurons']['gRaw'][AVAL,:])
    axava1.plot(time_contig, neurons_withNaN[AVAL,:], color='#1f77b4', label='AVAL')
    axava1.plot(time_contig, neurons_withNaN[AVAR, :], color='blue', label='AVAR')
    axava1.legend()
    axava1.set_xticks(np.arange(0, time_contig[-1], 60))
    axava1.set_ylabel('AVA')
    fig.colorbar(pos, ax=axava1)
    axava1.set_xlim(0, end_time)
    axava1.axes.get_xaxis().set_visible(False)





    #Repeat on the derivatives a la Kato et al
    def take_deriv(neurons):
        from prediction.Classifier import rectified_derivative
        _, _, nderiv = rectified_derivative(neurons)
        return nderiv
    Neuro_dFdt = take_deriv(neurons[valid_imm[idx_clust], :]).T
    Neuro = np.copy(neurons[valid_imm[idx_clust], ]).T  # I_smooth_interp_nonctoig



    def center_and_scale_around_immmobile_portion(recording, imm_start_index, end_index, with_std=False):
        # subtract and rescale the whole recording  so that the mean during hte immobile portion is zero
        # and, optionally, so that the variance during immobile portion is 1
        from sklearn.preprocessing import StandardScaler
        mean_scale = StandardScaler(copy=True, with_mean=True, with_std=with_std)
        mean_scale.fit(recording[imm_start_index:end_index, :]) #calcluate mean and or variance based on immobile
        return mean_scale.transform(recording) #rescale based on whole recording

    Neuro_mean_sub = center_and_scale_around_immmobile_portion(Neuro, imm_start_index, end_index, with_std=False)
    Neuro_z = center_and_scale_around_immmobile_portion(Neuro, imm_start_index, end_index, with_std=True)
    Neuro_dFdt_mean_sub = center_and_scale_around_immmobile_portion(Neuro_dFdt, imm_start_index, end_index, with_std=False)
    Neuro_dFdt_z = center_and_scale_around_immmobile_portion(Neuro_dFdt, imm_start_index, end_index, with_std=True)

    def print_AVAs_weights_in_pcs(AVAL_ci, AVAR_ci, pca, label=''):
        from sklearn.decomposition import PCA
        print(label)
        print("AVAL weights:", pca.components_[:, AVAL_ci])
        print("AVAR weights:", pca.components_[:, AVAR_ci])
        print("AVAL ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVAL_ci))
        print("AVAR ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVAR_ci))
        return

    def project_into_immobile_pcs(recording, imm_start_index, end_index, AVAL_ci=None, AVAR_ci=None, label=''):
        # Plot neural state space trajectories in first 3 PCs
        # also reduce dimensionality of the neural dynamics.
        from sklearn.decomposition import PCA
        nComp = 3  # pars['nCompPCA']
        pca = PCA(n_components=nComp, copy=True)
        pcs = pca.fit(recording[imm_start_index:end_index, :]).transform(recording)
        if AVAL_ci is not None and AVAR_ci is not None:
            print_AVAs_weights_in_pcs(AVAL_ci, AVAR_ci, pca, label=label)
        del pca
        return np.copy(pcs)


    pcs = project_into_immobile_pcs(Neuro_mean_sub, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='activity')
    pcs_z = project_into_immobile_pcs(Neuro_z, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='z-score')
    pcs_dFdt = project_into_immobile_pcs(Neuro_dFdt_mean_sub, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='deriv')
    pcs_dFdt_z = project_into_immobile_pcs(Neuro_dFdt_z, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='deriv Z-scored')
    

    theta, phi =None, None
    theta, phi = 83, 13
    plot_trajectories(pcs, time, drug_app_index, imm_start_index, im_end, key + '\n F  PCA (minimally processed)', theta=theta, phi=phi)
    plot_trajectories(pcs_z, time, drug_app_index, imm_start_index, im_end, key + '\n F  PCA (z-scored)', theta=theta, phi=phi)
    plot_trajectories(pcs_dFdt, time, drug_app_index, imm_start_index, im_end, key + '\n F  dF/dt PCA (minimally processed)', color="#ff7f0e", theta=theta, phi=phi)
    plot_trajectories(pcs_dFdt_z, time, drug_app_index, imm_start_index, im_end, key + '\n F  dF/dt PCA (z-scored)', color="#ff7f0e", theta=theta, phi=phi)

    offset = 20
    axpc = fig.add_subplot(gs[4, :], sharex=ax)
    axpc.plot(time, pcs_z[:, 0], label='PC1', color='#6339F6')
    axpc.plot(time, offset + pcs_z[:, 1], label='PC2', color='#426BEB')
    axpc.plot(time, 2 * offset + pcs_z[:, 2], label='PC3', color='#3DC1F6')
    axpc.set_xticks(np.arange(0, time_contig[-1], 60))
    axpc.legend()
    fig.colorbar(pos, ax=axpc)
    axpc.set_xlim(0, end_time)
    axpc.axes.get_xaxis().set_visible(False)

    offset = 24
    axpcdf = fig.add_subplot(gs[5, :], sharex=ax)
    axpcdf.plot(time, pcs_dFdt_z[:, 0], label='PC1 dF/dt', color='#F6C802')
    axpcdf.plot(time, offset + pcs_dFdt_z[:, 1], label='PC2 dF/dt', color='#EB9B0F')
    axpcdf.plot(time, 2 * offset + pcs_dFdt_z[:, 2], label='PC3 dF/dt', color='#F66607')
    axpcdf.set_xticks(np.arange(0, time_contig[-1], 60))
    axpcdf.legend()
    fig.colorbar(pos, ax=axpcdf)
    axpc.set_xlim(0, end_time)

    codePath = userTracker.codePath()
    outputFolder = os.path.join(codePath,'figures/subpanels_revision/generatedFigs')
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, key + "transition.pdf"))
    pdf.savefig(fig)
    pdf.close()




    ### Next it will be important to show that the neurons before and after transitions
    # are likely the same
    before = np.arange(700, 800)
    after = np.arange(1100, 1200)
    av_b = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, before[0]:before[-1]], axis=1)
    av_a = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, after[0]:after[-1]], axis=1)
    av_bprime = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, before[0] - 400 : before[-1] - 400], axis=1)
    av_aprime = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, after[0] + 400 : after[-1] + 400], axis=1)
    plt.figure()
    for k in np.arange(av_b.shape[0]):
        plt.plot([0, 1], [av_b[k], av_a[k]], 'ko-')
        plt.plot([3, 4], [av_bprime[k], av_b[k]], 'ko-')
        plt.plot([5, 6], [av_a[k], av_aprime[k]], 'ko-')
    plt.text(0, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_b - av_a)))
    plt.text(3, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_bprime - av_b)))
    plt.text(5, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_a - av_aprime)))
    labels = ['(700 to 800)', '(1100 to 1200)', '(300 to 400)', '(700 to 800)', '(1100 to 1200)', '(1500 to 1600)']
    plt.xticks([0, 1, 3, 4, 5, 6], labels)
    plt.title('Change in Mean raw RFP Values across different time windows')
    plt.ylabel('F')
    plt.xlabel('Averaging Window (Volumes)')

    diff = av_a - av_b
    print("Neurons that have RFP values that change a lott before and after transition. (top two increase; top two decrease; python indexing")
    print(valid_imm[np.argsort(diff)[[0, 1, -1, -2]]])


    ######################################################
    #  Now we are lookikng at correlation structure between moving and immobile
    #########################################################
    def get_pearsons_coefs(recording):
        cc = np.corrcoef(recording)
        indices = np.tril_indices(cc.shape[0],-1) # get the lower triangle excluding the diagonel
        return cc[indices], cc

    rho_mov, cmat_mov = get_pearsons_coefs(Neuro_mean_sub[:drug_app_index, :].T)
    rho_imm, cmat_imm = get_pearsons_coefs(Neuro_mean_sub[imm_start_index:, :].T)


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




    ########
    # Now generate correlation matrices
    ########

    def do_clustering(matrix):
        np.random.seed(seed=42)
        import scipy.cluster.hierarchy as sch
        Y = sch.linkage(matrix, method='single')
        Z = sch.dendrogram(Y, no_plot=True)
        return Z['leaves']

    def plot_corrMatrices(cmat_mov, cmat_imm, AVAL=None, AVAR=None):
        AVAL = np.squeeze(AVAL)
        AVAR = np.squeeze(AVAR)
        cfig = plt.figure()
        gs = gridspec.GridSpec(3, 3, figure=cfig)
        ax1 = cfig.add_subplot(gs[0, 0])
        ax2 = cfig.add_subplot(gs[0, 1])
        ax3 = cfig.add_subplot(gs[0, 2])
        ax4 = cfig.add_subplot(gs[1, 0])
        ax5 = cfig.add_subplot(gs[1, 1])
        ax6 = cfig.add_subplot(gs[1, 2])
        ax7 = cfig.add_subplot(gs[2, 0])
        ax8 = cfig.add_subplot(gs[2, 2])

        # Cluster based on moving
        cgIdx = do_clustering(cmat_mov)
        ax1.imshow(cmat_mov[:, cgIdx][cgIdx], vmin=-1, vmax=1)
        pos=ax2.imshow(cmat_imm[:, cgIdx][cgIdx], vmin=-1, vmax=1)
        pos2= ax3.imshow(cmat_imm[:, cgIdx][cgIdx] - cmat_mov[:, cgIdx][cgIdx], vmin=-1.3, vmax=1.35, cmap='plasma')

        cb = fig.colorbar(pos, ax=ax7)
        cb2 = fig.colorbar(pos2, ax=ax8)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()


        #Cluster based on immobile
        cgIdx_im = do_clustering(cmat_imm)
        ax4.imshow(cmat_mov[:, cgIdx_im][cgIdx_im], vmin=-1, vmax=1)
        ax5.imshow(cmat_imm[:, cgIdx_im][cgIdx_im], vmin=-1, vmax=1)
        ax6.imshow(cmat_imm[:, cgIdx_im][cgIdx_im] - cmat_mov[:, cgIdx_im][cgIdx_im], vmin=-1.3, vmax=1.35, cmap='plasma')

        N = cmat_imm.shape[0]
        axarray = [ax1, ax2, ax3, ax4, ax5, ax6]
        indxarray = [cgIdx, cgIdx, cgIdx, cgIdx_im, cgIdx_im, cgIdx_im]
        if AVAL is not None and AVAR is not None:
            for k in np.arange(len(axarray)):
                a = np.concatenate([np.where(indxarray[k] == AVAL), np.where(indxarray[k] == AVAR)])
                ticks = np.concatenate([np.squeeze(a), np.array([0, N-1])])
                axarray[k].set_yticks(ticks)
                axarray[k].set_yticklabels(['AVAL', 'AVAR', str(0), str(N-1)])
                axarray[k].set_xticks(ticks)
                axarray[k].set_xticklabels(['AVAL', 'AVAR', str(0), str(N-1)])


    plot_corrMatrices(cmat_mov, cmat_imm, AVAL=AVAL_ci, AVAR=AVAR_ci)
    print("Plotting.")

    def boxdotplot(combined_data, labels=[''], title='', ylabel='', swarm=True, ylims=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure()
        sns.set_style("white")
        sns.set_style("ticks")
        if swarm:
            ax = sns.swarmplot(data=combined_data, color="black")
            ax = sns.boxplot(data=combined_data,
                         showcaps=False, boxprops={'facecolor': 'None'},
                         showfliers=False, whiskerprops={'linewidth': 0})
        else:
            ax = sns.boxplot(data=combined_data, boxprops={'facecolor': 'None'})
        ax.set_xticklabels(labels)
        yloc = plt.MaxNLocator(5)
        ax.yaxis.set_major_locator(yloc)
        sns.despine()
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylims is not None:
            ax.set_ylim(ylims)



    def get_diff_dist(rhos, cum_prob):
        #calculate distribution of differences of correlations
        delta_rhos = rhos - rhos[np.random.permutation(rhos.size)] #randomly subtract one rho rom another
        delta_rhos = np.sort(delta_rhos)
        cum_prob_delta = np.linspace(0, 1, len(delta_rhos), endpoint=False)
        return delta_rhos, cum_prob_delta

    def get_sig_thresh(rhos, cum_prob, mult_hyp=1):
        sig = 0.05
        pos_thresh = np.min(rhos[cum_prob>1-sig/mult_hyp])
        neg_thresh = np.max(rhos[cum_prob<sig/mult_hyp])
        return neg_thresh, pos_thresh

    def get_pvals(vec, rhos, cum_prob):
        #Calcualte p-values for a vector of rhos
        pvals=np.zeros(vec.shape)
        from figures.subpanels_revision.tuning_of_highly_weighted_neurons import get_pval_from_cdf
        for k, p in enumerate(vec):
            pvals[k] = get_pval_from_cdf(p, rhos, cum_prob)
        return pvals

    # Inspect AVA's Correlations
    def compare_AVA_correlation(cmat_imm, cmat_mov, AVA_ci, AVA_other_ci,  rhos=None, cum_prob=None, title=''):
        AVA_ci = AVA_ci.flatten()
        AVA_other_ci = AVA_other_ci.flatten()

        corr_im = cmat_imm[:, AVA_ci].squeeze()
        corr_mv = cmat_mov[:, AVA_ci].squeeze()
        N = corr_im.size
        print(N)

        #Sort by strength of correlation to AVA during movement
        sidx_mv = np.flipud(np.argsort(corr_mv))

        # Calculate matrix of differenes between two corr matrices
        d = corr_im - corr_mv

        plt.figure()
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True)
        fig.suptitle(title)
        ax1.barh(np.arange(N), corr_mv[sidx_mv], align='center')
        ax2.barh(np.arange(N), corr_im[sidx_mv], align='center')
        ax3.barh(np.arange(N), d[sidx_mv], align='center')
        ax1.set_xlim([-1.05, 1.05])
        ax2.set_xlim([-1.05, 1.05])
        ax3.set_xlim([-1.05, 1.05])

        j=10
        diffsort = np.argsort(d[sidx_mv])
        topj = np.concatenate([diffsort[-j:],diffsort[:j]])
        for k in topj:
            ax3.text(np.sign(d[sidx_mv][k]), k, str(sidx_mv[k]))

        sig_diff = sidx_mv[topj] #convert from index locally storted
        if rhos is not None and cum_prob is not None:
            #find out which neurons have a change in correlation that pass  multi hyp tested threshold and flag them
            delta_rhos, cum_prob_delta =  get_diff_dist(rhos, cum_prob)
            #pvals = get_pvals(np.squeeze(d), delta_rhos, cum_prob_delta)

            #from statsmodels.stats.multitest import multipletests
            #reject, _, _, _ = multipletests(pvals) # uses holm-sidak adjustment
            #for k in np.arange(reject.size)[reject]:
            #    ax3.text(1.1*np.sign(d[k]), np.argwhere(sidx_mv==k)-.7, '*')

            #Alternatively use vanilla bonferonni correction
            nthresh, pthresh = get_sig_thresh(delta_rhos, cum_prob_delta, mult_hyp=N-1)
            ax3.axvline(nthresh, linestyle='dashed', color='red')
            ax3.axvline(pthresh, linestyle='dashed', color='red')
            sig_diff = []
            for k in np.arange(d.size):
                if d[k] > pthresh or d[k] < nthresh:
                    ax3.text(1.1*np.sign(d[k]), np.argwhere(sidx_mv==k)-.7, str(k))
                    sig_diff.append(k)

            nthresh, pthresh = get_sig_thresh(rhos, cum_prob, mult_hyp=N-1)
            ax2.axvline(nthresh, linestyle='dashed', color='red')
            ax2.axvline(pthresh, linestyle='dashed', color='red')
            ax1.axvline(nthresh, linestyle='dashed', color='red')
            ax1.axvline(pthresh, linestyle='dashed', color='red')


        ax1.set_xlabel('rho')
        ax2.set_xlabel('rho')
        ax3.set_xlabel('rho')
        ax1.set_title('Moving')
        ax2.set_title('Immobile')
        ax3.set_title('Immobile - Moving')
        ax1.set_yticks([np.where(sidx_mv==AVA_ci), np.where(sidx_mv==AVA_other_ci)])
        ax1.set_yticklabels(['This AVA', 'Other AVA'])
        return sidx_mv, sig_diff

    from figures.subpanels_revision.tuning_of_highly_weighted_neurons import shuffled_cdf_rho, get_pval_from_cdf


    def all_indices_but(length, excl):
        out = np.ones(length, dtype=bool)
        out[np.squeeze(excl)] = False
        return out

    N = cmat_imm.shape[0]
    #calculate distribution of correlations

    #print("calculating shuffles for AVAR..")
    #rhos, cum_prob = shuffled_cdf_rho(neurons[all_indices_but(N, AVAR_ci), :], np.squeeze(neurons[AVAR_ci, :]), None,
    #                                  nShuffles=2000, shuffle_phase=True)

    #neg_thresh, pos_thresh = get_sig_thresh(rhos, cum_prob, mult_hyp=N-1)
    #neg_delta_thresh, pos_delta_thresh = get_sig_thresh(delta_rhos, cum_prob_delta, mult_hyp=N-1)
    sidx_mv, topj_R = compare_AVA_correlation(cmat_imm, cmat_mov, AVAR_ci, AVAL_ci, rhos=None, cum_prob=None, title='AVAR')
    print("Neurons of interest in restricted clustered indexing:",topj_R, 'and in original full index', valid_imm[idx_clust[topj_R]])

    #print("calculating shuffles for AVAL")
    #rhos, cum_prob = shuffled_cdf_rho(neurons[all_indices_but(N, AVAL_ci), :], np.squeeze(neurons[AVAL_ci, :]), None,
    #                                 nShuffles=2000, shuffle_phase=True)
    sidx_mv, topj_L = compare_AVA_correlation(cmat_imm, cmat_mov, AVAL_ci, AVAR_ci,  rhos=None, cum_prob=None, title='AVAL')

    #Summary statistic
    alldata = [np.squeeze(cmat_mov[:, AVAL_ci]), np.squeeze(cmat_imm[:, AVAL_ci]),
               np.squeeze(cmat_mov[:, AVAR_ci]), np.squeeze(cmat_imm[:, AVAR_ci])]
    from scipy import stats
    _, p_L = stats.wilcoxon(np.squeeze(cmat_mov[:, AVAL_ci]), np.squeeze(cmat_imm[:, AVAL_ci]))
    _, p_R = stats.wilcoxon(np.squeeze(cmat_mov[:, AVAR_ci]), np.squeeze(cmat_imm[:, AVAR_ci]))
    boxdotplot(alldata, ['AVAL mov', 'AVAL imm', 'AVAR mov', 'AVAR imm'], 'Correlations to AVA, p= %.3f, %.3f' %(p_L, p_R) , 'rho')
    AVA_mov = np.squeeze(cmat_mov[:, [AVAL_ci, AVAR_ci]].ravel())
    AVA_imm = np.squeeze(cmat_imm[:, [AVAL_ci, AVAR_ci]].ravel())
    _, p_ava=stats.wilcoxon(AVA_mov, AVA_imm)
    boxdotplot([AVA_mov, AVA_imm], ['AVA mov', 'AVA imm'], 'Correlations to AVA, p= %.3f' % p_ava, 'rho', ylims=[-1.1, 1.1])



    HandPickedNeurons = True
    def ci(fulli):
        #Get the clustered index from the full index
        return np.squeeze(np.argwhere(valid_imm[idx_clust] == fulli))

    #Traces of Interest
    if HandPickedNeurons:
        #ofint = np.array([ci(78), ci(3), 33, 22, ci(AVAR), ci(AVAL),  ci(5), ci(47)]) #ci(76) is also good.
        ofint = np.array([ci(78),  33, ci(AVAR), ci(AVAL), ci(5), ci(47)])
    else:
        ofint = np.concatenate([np.squeeze(np.array([AVAL_ci, AVAR_ci])), np.union1d(topj_L, topj_R)])
    fig, axarr = plt.subplots(nrows=len(ofint), ncols=1, sharex=True)
    for k, noi in enumerate(ofint): #For each neuron of interest
        activity_oi = np.squeeze(neurons_withNaN[valid_imm[idx_clust[noi]], :])
        axarr[k].plot(time_contig, activity_oi, label='Neuron %d (%d)' % (noi, valid_imm[idx_clust[noi]]))
        axarr[k].legend()
        axarr[k].axvline(drug_app_time)
        axarr[k].axvline(imm_start_time)


    print("Plotting scatterplot..")
    fig, ax = plt.subplots()
    ax.plot(alldata[0], alldata[1], 's', label='AVAL', fillstyle='none', color="gray")
    ax.plot(alldata[2], alldata[3], 'o', label='AVAR', fillstyle='none')
    ax.legend()
    ax.axis('square')
    ax.set_aspect('equal')
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_xlabel('rho_AVA,j moving')
    ax.set_ylabel('rho_AVA,j immobile')
    for k, noi in enumerate(ofint): #For each neuron of interest
        noi = np.int(noi)
        ax.text(alldata[0][noi], alldata[1][noi], '%d(%d)' % (noi, valid_imm[idx_clust[noi]]))
        ax.text(alldata[2][noi], alldata[3][noi], '%d(%d)' % (noi, valid_imm[idx_clust[noi]]))

    from figures.subpanels_revision.compare_corr_structure_all_recordings import dissimilarity
    print(dissimilarity(cmat_mov, cmat_imm))
    plt.show()




print("Done")
