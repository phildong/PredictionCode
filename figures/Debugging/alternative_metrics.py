from scipy.signal import butter, lfilter

#Based on https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    ################################################
    #
    # grab all the data we will need
    #
    ################################################
    import os
    import numpy as np

    from prediction import userTracker
    import prediction.dataHandler as dh

    print("Loading Data...")

    data = {}
    # for typ in ['AML32', 'AML18', 'AML175', 'AML70']:
    for typ in ['AML32']:
        for condition in ['moving']:  # ['moving', 'immobilized', 'chip']:
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

    ### Get the relevant Data
    key = 'AML32_moving'
    idn = 'BrainScanner20170424_105620'
    behavior = 'AngleVelocity'


    moving = data[key]['input'][idn]
    valid_map = moving['Neurons']['I_valid_map']

    movingAnalysis = data[key]['analysis'][idn]
    splits = movingAnalysis['Training']
    train, test = splits[behavior]['Train'], splits[behavior]['Test']
    beh = moving['BehaviorFull'][behavior]
    time = moving['Neurons']['I_Time']
    time_crop_noncontig = moving['Neurons']['I_Time_crop_noncontig']

    behPred_SN = np.empty(moving['BehaviorFull'][behavior].shape)
    behPred_SN[:] = np.nan
    behPred_SLM = np.copy(behPred_SN)

    activity = moving['Neurons']['I_smooth_interp_crop_noncontig']
    beh_crop_noncontig = moving['Behavior_crop_noncontig'][behavior]

    #### Plot the unfiltered recording and SLM and Best Single Neuron Prediction
    print("Plotting the unfiltered recording and the predictions.")
    #### Find best neuron

    numNeurons = activity.shape[0]
    R2_local_all = np.empty(numNeurons)
    R2_local_all[:] = np.nan

    m = np.empty(numNeurons)
    b = np.empty(numNeurons)
    m[:] = np.nan
    b[:] = np.nan

    from inspectPerformance import linear_fit
    from inspectPerformance import calc_R2
    from inspectPerformance import linfunc

    # go through all the neurons in the recording
    for neuron in np.arange(numNeurons):
        # Perform a linear fit and extram m & b for y=mx+b
        m[neuron], b[neuron], pcov = linear_fit(activity[neuron, [train]].flatten(),
                                                beh_crop_noncontig[train])
        behPred_SN[:] = np.nan

        # using the m & b, generate the predicted behavior
        behPred_SN[valid_map] = linfunc(activity[neuron, :], m[neuron], b[neuron])

        # Evaluate how well we did at predicting the behavior
        R2_local_all[neuron] = calc_R2(beh[valid_map][test], behPred_SN[valid_map][test])

    # Store the best predicting neurons
    bestNeuron = np.argmax(R2_local_all)
    # regenerate our prediction from the best predicted neuron
    behPred_SN[valid_map] = linfunc(activity[bestNeuron, :], m[bestNeuron], b[bestNeuron])

    ### Get the behavior predicted from the best Senigle Neuron
    behPred_SLM[valid_map] = movingAnalysis['ElasticNet'][behavior]['output']


    import matplotlib.pyplot as plt

    fig=plt.figure(figsize=[24, 14])
    fig.suptitle('NO ADDITIONAL FILTERING data[' + key + '][' + idn + ']')
    row = 2
    col = 2

    R2 = calc_R2(beh[valid_map][test], behPred_SLM[valid_map][test])
    R2_train = calc_R2(beh[valid_map][train], behPred_SLM[valid_map][train])

    ax1 = fig.add_subplot(row, col, 1, title='SLM Unfiltered R2=%.2f, R_train = %.2f' % (R2, R2_train))
    ax1.plot(time, beh, label="Measured")
    ax1.plot(time, behPred_SLM, label="Predicted")
    ax1.axhline(linewidth=0.5, color='k')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity')
    ax1.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                                alpha=0.1)
    ax1.legend()


    ax3 = fig.add_subplot(row, col, 3, title='SLM Unfiltered Residuals R2=%.2f, R_train = %.2f' % (R2, R2_train), sharex=ax1, sharey=ax1)
    resid_SLM = beh-behPred_SLM
    ax3.plot(time, resid_SLM, 'g', label="resid")
    test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
    ax3.plot(time[test_indices],
             np.nancumsum((resid_SLM[test_indices]) ** 2) / np.nansum(
                 (beh[test_indices] - np.nanmean(beh[test_indices])) ** 2) * 3 * np.nanmean(beh ** 2),
             'm',
             label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')

    ax3.axhline(linewidth=0.5, color='k')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity')
    ax3.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                                alpha=0.1)
    ax3.legend()


    R2 = calc_R2(beh[valid_map][test], behPred_SN[valid_map][test])
    R2_train = calc_R2(beh[valid_map][train], behPred_SN[valid_map][train])


    ax2 = fig.add_subplot(row, col, 2, title='Best Single Neuron Unfiltered R2=%.2f, R_train = %.2f' % (R2, R2_train), sharex=ax1, sharey=ax1)
    ax2.plot(time, beh, label="Measured")
    ax2.plot(time, behPred_SN, label="Predicted")
    ax2.axhline(linewidth=0.5, color='k')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity')
    ax2.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                                alpha=0.1)
    ax2.legend()


    ax4 = fig.add_subplot(row, col, 4, title='Best Single Neuron Unfiltered Residuals R2=%.2f, R_train = %.2f' % (R2, R2_train), sharex=ax1, sharey=ax1)
    resid_SN = beh-behPred_SN
    ax4.plot(time, resid_SN, 'g', label="resid")
    test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
    ax4.plot(time[test_indices],
             np.nancumsum((resid_SN[test_indices])**2) / np.nansum((beh[test_indices] - np.nanmean(beh[test_indices]))**2) * 3 * np.nanmean(beh**2),
            'm',
             label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')
    ax4.axhline(linewidth=0.5, color='k')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity')
    ax4.legend()
    ax4.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                                alpha=0.1)

    from prediction.dataHandler import gauss_filterNaN







    ## Goal here is to try smoothing different amounts and measure
    ## performance on the smoothed versions
    ## I think we will not only want to plot the residuals of the testset
    ## But maybe also plot the difference between the true and the smooth

    vps = 6
    least_filtering = 1 * vps
    most_filtering = 10 * vps
    num_filters = 10

    for sigma in np.linspace(least_filtering, most_filtering, num_filters):
        fig = plt.figure(figsize=[24, 14])
        fig.suptitle('$\\sigma=%.2f$ s Gauss Smoothing data[' % np.true_divide(sigma, vps) + key + '][' + idn + ']')
        row = 2
        col = 2

        #Smooth all the data
        
        only_smooth_neural = True

        if only_smooth_neural:
            beh_smooth = np.copy(beh)
        else:
            beh_smooth = gauss_filterNaN(beh, sigma)
            beh_smooth[np.isnan(beh)] = np.nan

        behPred_SLM_smooth = gauss_filterNaN(behPred_SLM, sigma)
        behPred_SLM_smooth[np.isnan(behPred_SLM)] = np.nan
        
        behPred_SN_smooth = gauss_filterNaN(behPred_SN, sigma)
        behPred_SN_smooth[np.isnan(behPred_SN)] = np.nan





        R2 = calc_R2(beh_smooth[valid_map][test], behPred_SLM_smooth[valid_map][test])
        R2_train = calc_R2(beh_smooth[valid_map][train], behPred_SLM_smooth[valid_map][train])

        ax1 = fig.add_subplot(row, col, 1, title='SLM NEURAL ONLY Smoothed $\\sigma=%.2f$ s R2=%.2f, R_train = %.2f' % (np.true_divide(sigma, vps), R2, R2_train))
        ax1.plot(time, beh_smooth, label="Measured")
        ax1.plot(time, behPred_SLM_smooth, label="Predicted")
        ax1.axhline(linewidth=0.5, color='k')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity')
        ax1.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax1.legend()

        ax3 = fig.add_subplot(row, col, 3, title='SLM NEURAL ONLY Smoothed $\\sigma=%.2f$ s Residuals R2=%.2f, R_train = %.2f' % (np.true_divide(sigma, vps), R2, R2_train),
                              sharex=ax1, sharey=ax1)
        resid_SLM_smooth = beh_smooth - behPred_SLM_smooth
        ax3.plot(time, resid_SLM_smooth, 'g', label="resid")
        test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
        ax3.plot(time[test_indices],
                 np.nancumsum((resid_SLM_smooth[test_indices]) ** 2) / np.nansum(
                     (beh_smooth[test_indices] - np.nanmean(beh_smooth[test_indices])) ** 2) * 3 * np.nanmean(beh_smooth ** 2),
                 'm',
                 label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')

        ax3.axhline(linewidth=0.5, color='k')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity')
        ax3.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax3.legend()

        R2 = calc_R2(beh_smooth[valid_map][test], behPred_SN_smooth[valid_map][test])
        R2_train = calc_R2(beh_smooth[valid_map][train], behPred_SN_smooth[valid_map][train])

        ax2 = fig.add_subplot(row, col, 2,
                              title='Best Single Neuron NEURAL ONLY Smoothed $\\sigma=%.2f$ s R2=%.2f, R_train = %.2f' % (np.true_divide(sigma, vps), R2, R2_train),
                              sharex=ax1, sharey=ax1)
        ax2.plot(time, beh_smooth, label="Measured")
        ax2.plot(time, behPred_SN_smooth, label="Predicted")
        ax2.axhline(linewidth=0.5, color='k')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax2.legend()

        ax4 = fig.add_subplot(row, col, 4,
                              title='Best Single Neuron NEURAL ONLY Smoothed $\\sigma=%.2f$ s Residuals R2=%.2f, R_train = %.2f' % (np.true_divide(sigma, vps), R2, R2_train),
                              sharex=ax1, sharey=ax1)
        resid_SN_smooth = beh_smooth - behPred_SN_smooth
        ax4.plot(time, resid_SN_smooth, 'g', label="resid")
        test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
        ax4.plot(time[test_indices],
                 np.nancumsum((resid_SN_smooth[test_indices]) ** 2) / np.nansum(
                     (beh_smooth[test_indices] - np.nanmean(beh_smooth[test_indices])) ** 2) * 3 * np.nanmean(beh_smooth ** 2),
                 'm',
                 label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')
        ax4.axhline(linewidth=0.5, color='k')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity')
        ax4.legend()
        ax4.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)

        import prediction.provenance as prov
        prov.stamp(plt.gca(), .55, .15)

    print("Beginning to save smoothing metric plots")
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("smoothingmetric.pdf")
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved smoothing metric plots.")


    raise RuntimeError, "stop here for now."























    #Generating the fitlers
    print("Generating the filters..")
    from scipy.signal import freqz



    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 6.0
    lowest = 0.001
    highest = 1
    numBands = 3*10

    bandEdges = np.linspace(lowest, highest, numBands)

    order = 5


    # Plot the frequency response for a few different orders.
    filter_fig_num = plt.figure(figsize=[12,7]).number




    for lowcut, highcut in zip(bandEdges[0:-2], bandEdges[1:-1]):

        ### Create the filter
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2**13)

        ### Plot the filter shape
        plt.figure(filter_fig_num)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Bandpass = %.2f - %.2f" % (lowcut, highcut))

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best', prop={'size': 8})



        fig = plt.figure(figsize=[24, 14])
        fig.suptitle('Metric w/ Bandpass  %.2f - %.2f Hz filter data[' % (lowcut, highcut) + key + '][' + idn + ']')
        row = 3
        col = 2


        ##### Filter the true and predicted signals
        beh_filt = np.copy(beh)
        beh_filt[:] =np.nan
        behPred_SLM_filt = np.copy(beh_filt)
        behPred_SN_filt = np.copy(beh_filt)

        beh_filt[valid_map] = butter_bandpass_filter(beh[valid_map], lowcut, highcut, fs, order=order)
        behPred_SLM_filt[valid_map] = butter_bandpass_filter(behPred_SLM[valid_map], lowcut, highcut, fs, order=order)
        behPred_SN_filt[valid_map] = butter_bandpass_filter(behPred_SN[valid_map], lowcut, highcut, fs, order=order)



        R2 = calc_R2(beh[valid_map][test], behPred_SLM[valid_map][test])

        ax1 = fig.add_subplot(row, col, 1, title='SLM Unfiltered R2=%.2f' % R2)
        ax1.plot(time, beh, label="Measured")
        ax1.plot(time, behPred_SLM, label="Predicted")
        ax1.axhline(linewidth=0.5, color='k')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity')
        ax1.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax1.legend()

        R2 = calc_R2(beh_filt[valid_map][test], behPred_SLM_filt[valid_map][test])

        ax3 = fig.add_subplot(row, col, 3, title='SLM Filtered (%.2f-%.2f Hz) R2=%.2f' % (lowcut, highcut, R2))
        plt.autoscale(enable=True)
        ax3.plot(time, beh_filt, label="Measured Filtered")
        ax3.plot(time, behPred_SLM_filt, label="Predicted Filtered")
        ax3.axhline(linewidth=0.5, color='k')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity')
        ax3.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax3.legend()

        ax5 = fig.add_subplot(row, col, 5,
                              title='SLM Filtered (%.2f-%.2f Hz) Residuals R2=%.2f' % (lowcut, highcut, R2),
                              sharex=ax3, sharey=ax3)
        resid_SLM_filt = beh_filt - behPred_SLM_filt
        ax5.plot(time, resid_SLM_filt, 'g', label="resid")
        test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
        ax5.plot(time[test_indices],
                 np.nancumsum((resid_SLM_filt[test_indices]) ** 2) / np.nansum(
                     (beh_filt[test_indices] - np.nanmean(beh_filt[test_indices])) ** 2) * 3 * np.nanmean(beh_filt ** 2),
                 'm',
                 label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')

        ax5.axhline(linewidth=0.5, color='k')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity')
        ax5.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax5.legend()



        R2 = calc_R2(beh[valid_map][test], behPred_SN[valid_map][test])

        ax2 = fig.add_subplot(row, col, 2, title='Best Single Neuron Unfiltered R2=%.2f' % R2, sharex=ax3, sharey=ax3)
        ax2.plot(time, beh, label="Measured")
        ax2.plot(time, behPred_SN, label="Predicted")
        ax2.axhline(linewidth=0.5, color='k')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax2.legend()


        R2 = calc_R2(beh_filt[valid_map][test], behPred_SN_filt[valid_map][test])

        ax4 = fig.add_subplot(row, col, 4,
                              title='Best Single Neuron Filtered (%.2f - %.2f Hz)  R2=%.2f' % (lowcut, highcut, R2),
                              sharex=ax3, sharey=ax3)
        ax4.plot(time, beh_filt, label="Measured Filtered")
        ax4.plot(time, behPred_SN_filt, label="Predicted Filtered")
        ax4.axhline(linewidth=0.5, color='k')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity')
        ax4.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        ax4.legend()


        #CONTINUE HERE
        ax6 = fig.add_subplot(row, col, 6,
                              title='Best Single Neuron Filtered (%.2f-%.2f Hz) Residuals R2=%.2f' % (lowcut, highcut, R2),
                              sharex=ax3, sharey=ax3)
        resid_SN_filt = beh_filt - behPred_SN_filt
        ax6.plot(time, resid_SN_filt, 'g', label="resid")
        test_indices = np.arange(valid_map[test[0]], valid_map[test[-1]])
        ax6.plot(time[test_indices],
                 np.nancumsum((resid_SN_filt[test_indices]) ** 2) / np.nansum(
                     (beh_filt[test_indices] - np.nanmean(beh_filt[test_indices])) ** 2) * 3 * np.nanmean(beh_filt ** 2),
                 'm',
                 label=r'$\frac{\sum_0^i(y-\hat{y})^2}{\sum(y-\langle y \rangle)^2} * 3\langle y \rangle ^2$')
        ax6.axhline(linewidth=0.5, color='k')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Velocity')
        ax6.legend()
        ax6.axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                    alpha=0.1)
        plt.axis('tight')
        plt.show()





run()







