# Script to read in neuron data and do correlations
# First do correlation for all neuronxall neuron
# Using that matrix of data, pull out the data of AVA specific correlations

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16

# For data set 102246 (moving-immobile)-Moving 1-1590, Immobile 1700-4016 (maybe less/still immobile then), AVA 30 (and 18?)
# For data set 111126 (moving-immobile)-moving 1-1785, Immobile 1900-3360, AVA 28

import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh


def get_data(start_frame, end_frame, neurons, time, time_contig):
    # take the loaded in data and pull out the neuron data for moving or immobile.
    # look up time of start frame/end frame in the time_contig
    # see what index in time (noncontig) is closest to time_contig value
    start_time= time_contig[start_frame]
    start_index = np.abs(time - start_time).argmin()
    end_time = time_contig[end_frame]
    end_index = np.abs(time-end_time).argmin()
    neuron_data = neurons[:, start_index:end_index]
    [row, column] = neurons.shape
    neuron_num = list(range(0, row, 1))
    return neuron_data, neuron_num


def do_correlation_all(neuron_data):
    # take in the Ratio2 frames, do the correlations/make the matrix
    r = np.corrcoef(neuron_data)
    # square the matrix to get the r_square
    r_square = np.square(r)
    return r, r_square


def do_correlation_AVA(r, r_square, AVA_neuron):
    # take the big all correlations and find the AVA neuron
    AVA_correlation = r[AVA_neuron, :]
    AVA_r_square = r_square[AVA_neuron,:]
    return AVA_correlation, AVA_r_square

def do_residual(moving_worm, immobile_worm):
    # this function should take the residual of whatever we give it (should work for whole matrix correlation as well as AVA specific correlation)
    residual = moving_worm- immobile_worm
    return residual


for typ_cond in ['AKS297.51_moving']: #, 'AML32_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))
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
    for key in filter(lambda x: '110803' in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        # velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        # curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
        # For immobile- how is NaN neurons that are not hand tracked dealt with by the smooth_interp...

    moving_data, neuron_number = get_data(1, 1465, neurons, time, time_contig)
    moving_corr, moving_r_square = do_correlation_all(moving_data)
    AVA_corr, AVA_r_square  = do_correlation_AVA(moving_corr,moving_r_square, 33)
    # need a list of the number of neurons

    plot1 = plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(moving_corr)
    plt.colorbar()
    plt.title('Correlation_110803')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')

    plt.subplot(1,2,2)
    plt.imshow(moving_r_square)
    plt.colorbar()
    plt.title('R^2_110803')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')


    plot2 = plt.figure(2)
    plt.subplot(1,2,1)
    plt.bar(neuron_number, AVA_corr)
    plt.title('AVA Correlation_110803')
    plt.xlabel('Neuron')
    plt.ylabel('Correlation')

    plt.subplot(1,2,2)
    plt.bar(neuron_number, AVA_r_square)
    plt.title('AVA R^2_110803')
    plt.xlabel('Neuron')
    plt.ylabel('R^2')


    plt.show()

    # Want to plot all neuron correlation and AVA neuron correlation






