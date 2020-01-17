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
    behPred_SN = np.empty(moving['BehaviorFull'][behavior].shape)
    behPred_SN[:] = np.nan
    behPred_SLM = np.copy(behPred_SN)

    #### Find best neuron
    activity = moving['Neurons']['I_smooth_interp_crop_noncontig']
    beh_crop_noncontig = moving['Behavior_crop_noncontig'][behavior]

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

    fig=plt.figure(figsize=[24, 12])
    fig.suptitle('NO ADDITIONAL FILTERING data[' + key + '][' + idn + ']')
    row = 2
    col = 2


    ax1 = fig.add_subplot(row, col, 1, title='SLM Unfiltered')

    ax1.plot(time, beh, label="Measured")
    ax1.plot(time, behPred_SLM, label="Predicted")
    ax1.axhline(linewidth=0.5, color='k')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity')
    ax1.legend()


    ax3 = fig.add_subplot(row, col, 3, title='SLM Unfiltered Residuals', sharex=ax1, sharey=ax1)
    ax3.plot(time, beh-behPred_SLM, label="resid")
    ax3.axhline(linewidth=0.5, color='k')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity')
    ax3.legend()



    ax2 = fig.add_subplot(row, col, 2, title='Best Single NEuron Unfiltered')

    ax2.plot(time, beh, label="Measured")
    ax2.plot(time, behPred_SN, label="Predicted")
    ax2.axhline(linewidth=0.5, color='k')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity')
    ax2.legend()


    ax4 = fig.add_subplot(row, col, 4, title='Best Single Neuron Unfiltered Residuals', sharex=ax1, sharey=ax1)
    ax4.plot(time, beh-behPred_SN, label="resid")
    ax4.axhline(linewidth=0.5, color='k')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity')
    ax4.legend()



    raise RuntimeError, "Stop here"


    from scipy.signal import freqz



    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 6.0
    lowest = 0.001
    highest = fs/2
    numBands=4*10

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






        plt.figure()







    # Filter a noisy signal.
    T = 600
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 =0.2
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=1)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()







