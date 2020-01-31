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

from prediction import userTracker
import prediction.dataHandler as dh






def calc_R2(y_true, y_pred):
    """calculate the coefficient of determination
    as defined here:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

    The goal is to have a transparent external way of calculating R^2
    seperate of regressions, as a final sanity check.
    """

    #make sure that if there is a NaN in one set, there is a NaN in the
    # corresponding time point in the other
    y_true[np.isnan(y_pred)] = np.nan
    y_pred[np.isnan(y_true)] = np.nan

    u = np.nansum((y_true - y_pred) ** 2)
    v = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    R2 = (1 - u / v)
    return R2

def linfunc(x, m, b):
    return m * x + b

def linear_fit(activity, behavior):
    """Fit neural activity to behavior using a simple linear fit.
    """
    assert activity.ndim == 1, "linear fit funciton only works on a single neuron trace at a time"
    assert np.all(np.isfinite(activity)), "expects finite values"
    assert np.all(np.isfinite(behavior)), "expects finite values"

    #scale the behavior to something around 1 for ease of fitting
    #we will un-scale the results later
    beh_scalefactor = np.mean(behavior)
    beh = (1 / beh_scalefactor) * behavior

    # scale the activity to something around 1 for ease of fitting
    # we will un-scale the results later
    act_scalefactor = np.mean(activity)
    act = (1 / act_scalefactor) * activity

    from scipy.optimize import curve_fit

    bounds = ([-60, -60], # lower bound of m (slope) and  b offset
                [60, 60]) # upper bound of m(slope) and b offset

    popt_guess = [1, 0]  #as a first guess assume no slope and offset

    #y = mx + b where x is activity and y is behavior
    popt, pcov = curve_fit(linfunc, act, beh, p0=popt_guess, bounds=bounds)

    m = popt[0] * beh_scalefactor / act_scalefactor
    b = popt[1] * beh_scalefactor

    return m, b, pcov


def calc_weight_p_value(a, b):
    """ Calculate the p-value that the observed dot product of a and b do not come from the distribution
    of two random vectors in D dimensional space.

    See: https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions

    """

    D = a.shape[0] #dimensionality of the weights
    
    
    from numpy import linalg as la
    #Calculate normalized dot product
    normdot = np.dot(np.true_divide(a, la.norm(a)), np.true_divide(b, la.norm(b)))

    from scipy import special
    #rescale by dimensionality to set width of the gaussian distribution
    #And calculate the complementary error function to get a p-value
    return special.erfc(normdot / (1 / np.sqrt(D)))




def main():
    print("Inspecting performance...")

    data = {}
    for typ in ['AML32', 'AML18', 'AML175', 'AML70']:
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


    #Two goals to start wtih.
    #Goal 1: is to plot perforamance of each dataset for best single neuron, PCA and SLM
    #in such a way that we can do within dataset comparisons across models.
    #This should be a plot of 3point line-graphs, one for each model.

    import matplotlib.pyplot as plt

    fig=plt.figure(1,[12, 10])
    fig.suptitle('Performance on held out testset')
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    axes = [ax1, ax2]


    figSLM = plt.figure(figsize=[12,10])
    figSLM.suptitle('The alpha parameter found by the algorithm')
    axSLM1 = plt.subplot(1,2,1)
    axSLM2 = plt.subplot(1,2,2)
    axesSLM = [axSLM1, axSLM2]


    figSLMl1ratio = plt.figure(figsize=[12,10])
    figSLMl1ratio.suptitle('The l1ratio parameter found by the algorithm')
    axSLM1l1ratio = plt.subplot(1,2,1)
    axSLM2l1ratio = plt.subplot(1,2,2)
    axesSLMl1ratio = [axSLM1l1ratio, axSLM2l1ratio]





    b_cnt = 0
    titles = ['Velocity', 'Turning']
    labels = ['PCA', 'SLM', 'Best Neuron']

    for behavior in ['AngleVelocity', 'Eigenworm3']:
        scores = []
        #For each type of recording
        for key, marker in zip(['AML32_moving', 'AML70_chip', 'AML70_moving'],['o', "^", "^"]):
            dset = data[key]['analysis']
            results_from_one_dataset = []

            #For each recording
            for idn in dset.keys():

                results_pca = dset[idn]['PCAPred'][behavior]
                results_SLM = dset[idn]['ElasticNet'][behavior]
                try:
                    SLM_score = results_SLM['scorepredicted']
                    SLM_alpha = results_SLM['alpha']
                    SLM_l1_ratio = results_SLM['l1_ratio']
                except:
                    SLM_score = np.nan
                    SLM_alpha = np.nan




                try:
                    BSN_score = np.max(results_SLM['individualScore'])
                except:
                    BSN_score = np.nan
                results_from_one_dataset = np.array([results_pca['scorepredicted'], SLM_score, BSN_score])

                #Plot
                axes[b_cnt].plot(np.arange(0,len(results_from_one_dataset)), results_from_one_dataset, marker=marker, label=key + idn)

                axes[b_cnt].title.set_text(titles[b_cnt])
                axes[b_cnt].set_ylim([0, 0.7])
                axes[b_cnt].set_xticks(np.arange(0,len(results_from_one_dataset)))
                axes[b_cnt].set_xticklabels(labels)
                axes[b_cnt].legend(prop={'size': 8})
                axes[b_cnt].set_ylabel('R^2')

                if np.isfinite(SLM_score):
                    axesSLM[b_cnt].plot(SLM_score, SLM_alpha, 'o', label=key + idn)
                    axesSLM[b_cnt].legend(prop={'size': 9})
                    axesSLM[b_cnt].set_ylabel('Alpha')
                    axesSLM[b_cnt].set_xlabel('R^2')
                    axesSLM[b_cnt].set_xlim([0, 1])
                    axesSLM[b_cnt].title.set_text(titles[b_cnt])

                if np.isfinite(SLM_l1_ratio):
                    axesSLMl1ratio[b_cnt].plot(SLM_score, SLM_l1_ratio, 'o', label=key + idn)
                    axesSLMl1ratio[b_cnt].legend(prop={'size': 9})
                    axesSLMl1ratio[b_cnt].set_ylabel('l1_ratio')
                    axesSLMl1ratio[b_cnt].set_xlabel('R^2')
                    axesSLMl1ratio[b_cnt].title.set_text(titles[b_cnt])
                    axesSLMl1ratio[b_cnt].set_xlim([0, 1])





        b_cnt = b_cnt+ 1


    import prediction.provenance as prov
    prov.stamp(ax2,.55,.15)
    prov.stamp(axSLM1,.55,.15)
    prov.stamp(axSLM1l1ratio,.55,.15)

    #Goal 2: show the predictions explicilty for all recordings as compared to true
    # here we will probably have to generate PDFs or PPTs or somethign
    # because I assume its a lot of plots

    #Actually plot each prediction
    print("Plotting behavior predictions..")
    # For each type of recording
    for key in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML18_moving']:
        dset = data[key]['input']

        # For each recording
        for idn in dset.keys():
            # Plot with time traces of predicted and measured
            fig = plt.figure(figsize=[30, 12])
            fig.suptitle('data[' + key + '][' + idn + ']')

            row = 3
            col = 2
            axes = []
            for each in np.reshape(np.arange(row * col)+1, ( col, row), order='F').ravel():
                axes = np.append(axes, plt.subplot(row, col, each))
            fig.suptitle(' data[' + key + '][' + idn + ']')

            # Plot Showing Neural Weights
            fig_weights = plt.figure(figsize=[28, 12])
            fig_weights.suptitle('Neural weights data[' + key + '][' + idn + ']')
            #need one subplot for each column (behavior)
            axes_weights = []
            #setup the subplots so that by simply incrementing the index of
            # axes_weights we can move through all the rows of each column
            weightfig_rows = 2
            for each in np.reshape(np.arange(weightfig_rows * col)+1,(weightfig_rows,col), order='F').ravel():
                    axes_weights = np.append(axes_weights, plt.subplot(2, col, each))



            #Scatter plots of predicted vs Measured
            fig_scatter = plt.figure(figsize=[28, 12])
            fig_scatter.suptitle('Scatter  data[' + key + '][' + idn + ']')
            axes_scatter = []
            for each in np.arange(row * col)+1:
                axes_scatter = np.append(axes_scatter, plt.subplot(row, col, each))

            ax_cnt = -1
            for behavior, title, beh_cnt in zip(['AngleVelocity', 'Eigenworm3'], ['Velocity', 'Turn'], np.arange(2)):
                for flag, pred_type in zip(['PCAPred', 'ElasticNet', 'ElasticNet'], ['PCA', 'SLM', 'Best Neuron']):

                    ax_cnt = ax_cnt + 1


                    #Get the data
                    moving = data[key]['input'][idn]
                    valid_map = moving['Neurons']['I_valid_map']
                    movingAnalysis = data[key]['analysis'][idn]

                    splits = movingAnalysis['Training']

                    indices_contig_trace_test = valid_map[splits[behavior]['Test']]
                    train, test = splits[behavior]['Train'], splits[behavior]['Test']

                    beh = moving['BehaviorFull'][behavior]

                    time = moving['Neurons']['I_Time']
                    time_crop_noncontig = moving['Neurons']['I_Time_crop_noncontig']

                    behPred = np.empty(moving['BehaviorFull'][behavior].shape)
                    behPred[:] = np.nan
                    additional_title_text = ''

                    R2_stored = movingAnalysis[flag][behavior]['scorepredicted']

                    activity = moving['Neurons']['I_smooth_interp_crop_noncontig']
                    beh_crop_noncontig = moving['Behavior_crop_noncontig'][behavior]
                    if pred_type == 'Best Neuron':
                        #Get ready to test the best Neuron

                        numNeurons = activity.shape[0]
                        R2_local_all = np.empty(numNeurons)
                        R2_local_all[:] = np.nan

                        m = np.empty(numNeurons)
                        b = np.empty(numNeurons)
                        m[:] = np.nan
                        b[:] = np.nan
                        # go through all the neurons in the recording
                        for neuron in np.arange(numNeurons):
                            #Perform a linear fit and extram m & b for y=mx+b
                            m[neuron], b[neuron], pcov = linear_fit(activity[neuron, [train]].flatten(),
                                                                    beh_crop_noncontig[train])
                            behPred[:] = np.nan

                            #using the m & b, generate the predicted behavior
                            behPred[valid_map] = linfunc(activity[neuron, :], m[neuron], b[neuron])

                            #Evaluate how well we did at predicting the behavior
                            R2_local_all[neuron] = calc_R2(beh[valid_map][test], behPred[valid_map][test])

                        #Store the best predicting neurons
                        bestNeuron = np.argmax(R2_local_all)
                        #regenerate our prediction from the best predicted neuron
                        behPred[valid_map] = linfunc(activity[bestNeuron, :], m[bestNeuron], b[bestNeuron])


                        #Load the stored R2 value (just to see if its the same)
                        try:
                            R2_stored = np.max(movingAnalysis[flag][behavior]['individualScore'])
                        except:
                            R2_stored = np.nan



                        additional_title_text = ", Neuron ID: %i" % bestNeuron


                    else:
                        behPred[valid_map] = movingAnalysis[flag][behavior]['output']

                    R2_local = calc_R2(beh[valid_map][test], behPred[valid_map][test])
                    R2_train = calc_R2(beh[valid_map][train], behPred[valid_map][train])

                    #Actually Plot
                    axes[ax_cnt].plot(time, beh, label="Measured")
                    axes[ax_cnt].plot(time, behPred, label="Predicted")
                    #axes[ax_cnt].plot(time, beh-behPred, label="resid", linewidth=.5)

                    axes[ax_cnt].title.set_text(pred_type +  ', '
                                                + title + additional_title_text +
                                                ' R2 = %.2f' % R2_stored
                                                + ' R2_loc = %.2f' % R2_local +
                                                ' R2_train = %.2f' % R2_train)
                    axes[ax_cnt].legend()
                    axes[ax_cnt].set_xlabel('Time (s)')
                    axes[ax_cnt].set_xlim( [time[valid_map[0]], time[valid_map[-1]]])

                    axes[ax_cnt].axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                                alpha=0.1)


                    #Generate scatter plots too of prediction vs measured (on the other figure)
                    axes_scatter[ax_cnt].plot(beh[valid_map][train], behPred[valid_map][train], linestyle='',
                                              marker='o', markersize=0.7, label='Train')
                    axes_scatter[ax_cnt].plot(beh[valid_map][test], behPred[valid_map][test], linestyle='',
                                              marker='o', markersize=0.7, color='m', label='Test')
                    axes_scatter[ax_cnt].set_xlim([np.nanmin(beh), np.nanmax(beh)])
                    axes_scatter[ax_cnt].set_ylim([np.nanmin(beh), np.nanmax(beh)])

                    axes_scatter[ax_cnt].title.set_text(pred_type + ', '
                                                + title + additional_title_text +
                                                ' R2 = %.2f' % R2_stored
                                                + ' R2_loc = %.2f' % R2_local +
                                                ' R2_train = %.2f' % R2_train)
                    axes_scatter[ax_cnt].set_xlabel('Measured')
                    axes_scatter[ax_cnt].set_ylabel('Predicted')
                    axes_scatter[ax_cnt].plot([np.nanmin(beh), np.nanmax(beh)], [np.nanmin(beh), np.nanmax(beh)], 'r')
                    axes_scatter[ax_cnt].axhline(linewidth=0.5, color='k')
                    axes_scatter[ax_cnt].axvline(linewidth=0.5, color='k')
                    axes_scatter[ax_cnt].set_aspect('equal', 'box')
                    axes_scatter[ax_cnt].legend()


                    ax_weight_cnt = beh_cnt * 2
                    # Generate plots of the neural  weights
                    if pred_type == 'Best Neuron':
                        axes_weights[ax_weight_cnt].plot(bestNeuron, m[bestNeuron], marker='o', label='Best Single Neuron')

                    elif pred_type == 'SLM':
                        #For now we will just plot SLM weights raw without any scaling corrections for the variance
                        SLM_w_raw = movingAnalysis[flag][behavior]['weights']
                        SLM_w = np.true_divide(SLM_w_raw, np.std(activity,axis=1)) * np.std(beh_crop_noncontig)
                        axes_weights[ax_weight_cnt].plot(SLM_w_raw * np.linalg.norm(SLM_w)/np.linalg.norm(SLM_w_raw), label='SLM uncorrected * c')
                        axes_weights[ax_weight_cnt].plot(SLM_w, label='SLM')



                    elif pred_type == 'PCA':
                        meta_weights = movingAnalysis[flag][behavior]['weights']
                        pc_weights = movingAnalysis[flag][behavior]['PCA_components']

                        ## Check size and such
                        PCA_model_weights_raw = np.matmul(meta_weights, pc_weights)

                        # Correct for scaling done on behavior (no scaling on neural activity)
                        PCA_model_weights = PCA_model_weights_raw * np.std(beh_crop_noncontig)
                        axes_weights[ax_weight_cnt].plot(PCA_model_weights, label='PCA')


                axes_weights[ax_weight_cnt].set_xlabel('Neuron')
                axes_weights[ax_weight_cnt].set_ylabel('Weight')
                axes_weights[ax_weight_cnt].title.set_text(behavior +
                                                           ' weights. SLM to PCA p-value = %.3f'
                                                           % calc_weight_p_value(SLM_w, PCA_model_weights))
                axes_weights[ax_weight_cnt].legend()




                ax_weight_cnt = beh_cnt * 2 + 1
                #Now in the second row of the  weights figures
                # we want to plot the mean value of the nueral activity and the variance
                axes_weights[ax_weight_cnt].clear()
                axes_weights[ax_weight_cnt].errorbar(np.arange(activity.shape[0]),
                                                     np.nanmean(activity, axis=1),
                                                     yerr=np.nanstd(activity,axis=1)/2,
                                                     label='Mean Activity I and Standard Dev')
                axes_weights[ax_weight_cnt].set_xlabel('Neuron')
                axes_weights[ax_weight_cnt].set_ylabel('Fluorescent Intensity, I (common-noise removed)')
                axes_weights[ax_weight_cnt].legend()




            #Stamp the weights with git info
            prov.stamp(axes_scatter[ax_cnt], .55, .15)
            prov.stamp(axes[ax_cnt], .55, .15)
            prov.stamp(axes_weights[beh_cnt], .55, .15)

            # Generate Scatter plot to test how related velocity and body curvature are to one another
            vel = moving['BehaviorFull']['AngleVelocity']
            curve = moving['BehaviorFull']['Eigenworm3']
            fig_vc, ax_vc = plt.subplots(1,1, figsize=[8,8])
            fig_vc.suptitle('Measured Velocity vs Measured Curvature \n data[' + key + '][' + idn + ']')
            ax_vc.plot(vel, curve, linestyle='', marker='o', markersize=0.7)
            ax_vc.set_xlim([np.nanmin(vel), np.nanmax(vel)])
            ax_vc.set_ylim([np.nanmin(curve), np.nanmax(curve)])
            ax_vc.set_xlabel('Measured Velocity')
            ax_vc.set_ylabel('Mesaured Curvature')
            prov.stamp(ax_vc, .55, .15)




    print("Saving behavior predictions to pdf...")

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("prediction_performance.pdf")
    numFigs = plt.gcf().number + 1
    for fig in xrange(1, numFigs): ## will open an empty extra figure :(
        print("Saving Figure %d of %d" % (fig, numFigs))
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved.")



    raise RuntimeError, "Stopping here for now."

    ### Plot Heatmap for each recording


    ### Plot Neural State Space Trajectories

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("Plotting heatmaps.....")
    for key in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML32_immobilized', 'AML70_immobilized', 'AML18_moving', 'AML18_immobilized']:
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

            Iz = dset['Neurons']['ActivityFull']
            time = dset['Neurons']['I_Time']

            prcntile = 99.7
            fig = plt.figure(figsize=(22,12))
            plt.suptitle('data[' + key + '][' + idn + ']')


            ax = plt.subplot(4,1,1)
            pos = ax.imshow(I_smooth, aspect='auto',
                            interpolation='none', vmin=np.nanpercentile(I_smooth.flatten(), 0.1), vmax=np.nanpercentile(I_smooth.flatten(), prcntile),
                            extent = [ time[0], time[-1], 0, I_smooth.shape[0] ], origin='lower')
            ax.set_title('I_smooth_interp_noncontig  (smooth, common noise rejected, w/ NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)
            fig.tight_layout(rect=[0, 0.03, 1.1, 0.97])


            ax = plt.subplot(4,1,2)
            pos = ax.imshow(I_smooth_interp_crop_noncontig_wnans, aspect='auto',
                            interpolation='none', vmin=np.nanpercentile(I_smooth_interp_crop_noncontig_wnans,0.1), vmax=np.nanpercentile(I_smooth_interp_crop_noncontig_wnans.flatten(), prcntile),
                            extent = [ time[0], time[-1], 0, I_smooth_interp_crop_noncontig_wnans.shape[0] ], origin='lower')
            ax.set_title('I_smooth_interp_crop_noncontig_wnans  (smooth,  interpolated, common noise rejected, w/ large NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)


            ax = plt.subplot(4, 1, 3)
            pos = ax.imshow(Iz, aspect='auto',
                            interpolation='none', vmin=-2, vmax=2,
                            extent = [time[0], time[-1], 0, Iz.shape[0] ], origin='lower')
            ax.set_title('Activity  (per-neuron z-scored,  aggressive interpolation, common noise rejected,  Jeffs photobleach correction)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)

            beh = dset['BehaviorFull']['AngleVelocity']
            time = dset['Neurons']['I_Time']

            axbeh = plt.subplot(4, 1, 4)
            axbeh.plot(time, beh)
            fig.colorbar(pos, ax=axbeh)
            axbeh.axhline(linewidth=0.5, color='k')
            axbeh.set_ylim([-2, 4])
            axbeh.set_xlim(ax.get_xlim())

            axbeh.set_title('Velocity')
            axbeh.set_xlabel('Time (s)')
            axbeh.set_ylabel('Velocity (Body bends)')

            prov.stamp(plt.gca(),.9,.15)









    print("Beginning to save heat maps")
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("heatmaps.pdf")
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved heatmaps.")






    ### Plot Neural State Space Trajectories

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("Beginning to plot neural state space trajectories..")
    for key in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML32_immobilized', 'AML70_immobilized', 'AML18_moving', 'AML18_immobilized']:
        dset = data[key]['input']
        # For each recording
        for idn in dset.keys():

            dset = data[key]['input'][idn]

            # Plot neural state space trajectories in first 3 PCs
            # also reduce dimensionality of the neural dynamics.
            nComp = 3  # pars['nCompPCA']
            pca = PCA(n_components=nComp)
            Neuro = np.copy(dset['Neurons']['I_smooth_interp_crop_noncontig']).T

            # make sure data is centered
            sclar = StandardScaler(copy=True, with_mean=True, with_std=False)
            zscore = StandardScaler(copy=True, with_mean=True, with_std=True)
            Neuro_mean_sub = sclar.fit_transform(Neuro)
            Neuro_z = zscore.fit_transform(Neuro)

            pcs = pca.fit_transform(Neuro_mean_sub)
            pcs_z = pca.fit_transform(Neuro_z)

            fig = plt.figure(figsize=(12,8))
            plt.suptitle( key + idn  + '\n PCA (minimally processed)')
            for nplot in np.arange(6)+1:
                ax = plt.subplot(2,3,nplot, projection='3d')
                ax.plot(pcs[:,0], pcs[:,1], pcs[:,2])
                ax.view_init(np.random.randint(360), np.random.randint(360))
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
            prov.stamp(plt.gca(),.55,.15)


            fig = plt.figure(figsize=(12,8))
            plt.suptitle(key + idn + '\n PCA (z-scored) ')
            for nplot in np.arange(6)+1:
                ax = plt.subplot(2,3,nplot, projection='3d')
                ax.plot(pcs_z[:,0], pcs_z[:,1], pcs_z[:,2],color='orange')
                ax.view_init(np.random.randint(360), np.random.randint(360))
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
            prov.stamp(plt.gca(), .55, .15)



    print("Beginning to save state space trajectories")
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("moving_statespace_trajectories.pdf")
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved state space trajectories.")

if __name__ == '__main__':
    main()