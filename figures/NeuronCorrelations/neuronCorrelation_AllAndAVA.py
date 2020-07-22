# Script to read in neuron data and do correlations
# First do correlation for all neuronxall neuron
# Using that matrix of data, pull out the data of AVA specific correlations

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16

# For data set 102246 (moving-immobile)-Moving 1-1590, Immobile 1700-4016 (maybe less/still immobile then), AVA 29 (30 matlab) (and 17?)
# For data set 111126 (moving-immobile)-moving 1-1785, Immobile 1900-3245, AVA 27 (28 matlab)

import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap


def get_data(start_frame, end_frame, neurons, neurons_withNaN, time, time_contig):
    # take the loaded in data and pull out the neuron data for moving or immobile.
    # look up time of start frame/end frame in the time_contig
    # see what index in time (noncontig) is closest to time_contig value
    start_time= time_contig[start_frame]
    start_index = np.abs(time - start_time).argmin()
    end_time = time_contig[end_frame]
    end_index = np.abs(time-end_time).argmin()
    neuron_data = neurons[:, start_index:end_index]
    neuron_NaN_data = neurons_withNaN[:, start_index:end_index]
    [row, column] = neurons.shape
    neuron_num = list(range(0, row, 1))
    return neuron_data, neuron_NaN_data, neuron_num


def do_correlation_all(neuron_data):
    # take in the Ratio2 frames, do the correlations/make the matrix
    r = np.corrcoef(neuron_data)
    # square the matrix to get the r_square
    r_square = np.square(r)
    return r, r_square


def find_NaN_neurons(neuron_NaN_data, neuron_num):
    # step through each row of data and find the sum of NaNs. If majority NaN, save that index (neuron number)
    index_list = []
    for i in neuron_num:
        row = neuron_NaN_data[i,:]  # just pull out the row we want, then check if it is majority NaNs
        column = len(row)
        numNan = np.sum(np.isnan(row))
        ratio = float(numNan)/float(column)
        if ratio > 0.8:  # for the non-tracked neurons, all time points should be NaN, so ratio should be 1
            index_list.append(i)
    return index_list


def replace_NaN_neurons(r, index_list, neuron_num):
    # for the moving-immobile transition, use index list to see where non tracked neurons are. Replace those
    # rows/columns with 0s in the r/r_square
    for i in index_list:
        # each index in the list is a row of NaNs. replace the correlation with 0s if neuron isn't tracked
        r[i, :] = np.zeros(len(neuron_num))
        r[:, i] = np.zeros(len(neuron_num))
    return r


def do_correlation_AVA(r, r_square, AVA_neuron):
    # take the big all correlations and find the AVA neuron
    AVA_correlation = r[AVA_neuron, :]
    AVA_r_square = r_square[AVA_neuron,:]
    return AVA_correlation, AVA_r_square

def replace_Nan_AVA(AVA_list, index_list):
    for i in index_list:
        AVA_list[i] = 0
    return AVA_list


def do_residual(moving_worm, immobile_worm):
    # this function should take the residual of whatever we give it (should work for whole matrix correlation as well
    # as AVA specific correlation)
    residual = moving_worm- immobile_worm
    return residual


def do_clustering(matrix):
    cg = clustermap(matrix)
    cgIdx = np.array(cg.data2d.columns)
    return cgIdx


def cluster_matrix(matrix, cgIdx):
    clustered_1 = matrix[:, cgIdx]
    clustered_matrix = clustered_1[cgIdx, :]
    return clustered_matrix


# Want to cluster the calcium signal- use correlation clustering?
# sort the NaN data?
def cluster_calcium(neurons_NaN, cgIdx):
    cluster_neuron = neurons_NaN[cgIdx, :]
    return cluster_neuron

for typ_cond in ['AKS297.51_transition']: #, 'AKS297.51_moving']:
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
    for key in filter(lambda x: '102246' in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_withNaN = dataSets[key]['Neurons']['I_smooth'] # use this to find the untracked neurons after transition
        # velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        # curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
        # For immobile- how is NaN neurons that are not hand tracked dealt with by the smooth_interp...
        # Still do the correlation with all (the interpolated ones too, but then replace with 0s)?

    moving_data, moving_withNaN, neuron_number = get_data(1, 1590, neurons, neurons_withNaN, time, time_contig)
    moving_corr, moving_r_square = do_correlation_all(moving_data)
    AVA_corr_noNan, AVA_r_square_noNan = do_correlation_AVA(moving_corr,moving_r_square, 29)

    # cgIdx = do_clustering(moving_corr_NaN) cluster based on immobile

    immobile_data, data_withNaN, neuron_number_im = get_data(1700, 4016, neurons, neurons_withNaN, time, time_contig)
    immobile_corr, immobile_r_square = do_correlation_all(immobile_data)
    AVA_corr_im, AVA_r_square_im = do_correlation_AVA(immobile_corr, immobile_r_square, 29)
    nan_index = find_NaN_neurons(data_withNaN, neuron_number)
    immobile_corr_NaN = replace_NaN_neurons(immobile_corr, nan_index, neuron_number)
    immobile_r_NaN = replace_NaN_neurons(immobile_r_square, nan_index, neuron_number)

    moving_corr_NaN = replace_NaN_neurons(moving_corr, nan_index, neuron_number)
    moving_r_square_NaN = replace_NaN_neurons(moving_r_square, nan_index, neuron_number)

    #Put NaN/0s into the AVA measurements as well
    AVA_corr = replace_Nan_AVA(AVA_corr_noNan, nan_index)
    AVA_r_square = replace_Nan_AVA(AVA_r_square_noNan, nan_index)



    cgIdx_immobile = do_clustering(immobile_corr_NaN)
    clustered_immobile_corr = cluster_matrix(immobile_corr_NaN, cgIdx_immobile)  # Why can't I do this in one line? or is there a better way to do this?
    clustered_immobile_r = cluster_matrix(immobile_r_NaN, cgIdx_immobile)  # Why can't I do this in one line? or is there a better way to do this?

    clustered_moving_corr = cluster_matrix(moving_corr_NaN, cgIdx_immobile)
    clustered_moving_r = cluster_matrix(moving_r_square_NaN, cgIdx_immobile)

    # cluster the neuron signal
    cluster_immobile_neuron = cluster_calcium(data_withNaN, cgIdx_immobile)
    cluster_moving_neuron = cluster_calcium(moving_withNaN, cgIdx_immobile)

    # calculate residuals
    corr_residual = do_residual(clustered_moving_corr, clustered_immobile_corr)
    r_square_residual = do_residual(clustered_moving_r, clustered_immobile_r)
    ava_corr_residual = do_residual(AVA_corr, AVA_corr_im)
    ava_r_residual = do_residual(AVA_r_square, AVA_r_square_im)




# do the clustering on moving data
    cgIdx_moving = do_clustering(moving_corr_NaN)
    moving_cluster_moving_corr = cluster_matrix(moving_corr_NaN, cgIdx_moving)
    moving_cluster_immobile_corr = cluster_matrix(immobile_corr_NaN, cgIdx_moving)
    moving_cluster_moving_r = cluster_matrix(moving_r_square_NaN, cgIdx_moving)
    moving_cluster_immobile_r = cluster_matrix(immobile_r_NaN, cgIdx_moving)

    moving_corr_residual = do_residual(moving_cluster_moving_corr, moving_cluster_immobile_corr)
    moving_r_residual = do_residual(moving_cluster_moving_r, moving_cluster_immobile_r)

    moving_cluster_immobile_neuron = cluster_calcium(data_withNaN, cgIdx_moving)
    moving_cluster_moving_neuron = cluster_calcium(moving_withNaN, cgIdx_moving)

    plot1 = plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(clustered_moving_corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_102246_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,2)
    plt.imshow(clustered_immobile_corr,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_102246_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,3)
    plt.imshow(corr_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_111126')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plot2 = plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(clustered_moving_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R^2_102246_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,2)
    plt.imshow(clustered_immobile_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R^2_102246_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,3)
    plt.imshow(r_square_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_111126')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

#    plot3 = plt.figure(3)
#   plt.subplot(1,3,1)
#    plt.bar(neuron_number, AVA_corr)
#    plt.title('AVA Correlation_111126_moving')
#    plt.xlabel('Neuron')
#    plt.ylabel('Correlation')
#    plt.ylim(-0.8, 1)
#    plt.grid()

#    plt.subplot(1,3,2)
#    plt.bar(neuron_number, AVA_corr_im)
#    plt.title('AVA Correlation_111126_immobile')
#    plt.xlabel('Neuron')
#    plt.ylabel('Correlation')
#    plt.ylim(-0.8, 1)
#    plt.grid()

#    plt.subplot(1,3,3)
#    plt.bar(neuron_number, ava_corr_residual)
#    plt.title('Residual_moving-immobile_111126')
#    plt.xlabel('Neuron')
#    plt.ylabel('Residual')
#    #plt.ylim(-0.6, 1)
#    plt.grid()


#   plot4 = plt.figure(4)
#    plt.subplot(1,3,1)
#    plt.bar(neuron_number, AVA_r_square)
#    plt.title('AVA R^2_111126_moving')
#    plt.xlabel('Neuron')
#    plt.ylabel('R^2')
#    plt.ylim(0, 1)
#    plt.grid()

#    plt.subplot(1,3,2)
#    plt.bar(neuron_number, AVA_r_square_im)
#    plt.title('AVA R^2_111126_immobile')
#    plt.xlabel('Neuron')
#    plt.ylabel('R^2')
#    plt.ylim(0, 1)
#    plt.grid()

#    plt.subplot(1,3,3)
#    plt.bar(neuron_number, ava_r_residual)
#    plt.title('Residual_moving-immobile_111126')
#    plt.xlabel('Neuron')
#    plt.ylabel('Residual')
#    #plt.ylim(0, 1)
#    plt.grid()
    #######


    plot5 = plt.figure(5)
    plt.subplot(1,3,1)
    plt.imshow(moving_cluster_moving_corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_102246_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,2)
    plt.imshow(moving_cluster_immobile_corr,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_102246_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,3)
    plt.imshow(moving_corr_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_102246')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plot6 = plt.figure(6)
    plt.subplot(1,3,1)
    plt.imshow(moving_cluster_moving_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R_Square_102246_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,2)
    plt.imshow(moving_cluster_immobile_r,vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R_Square_102246_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plt.subplot(1,3,3)
    plt.imshow(moving_r_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_102246')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plot7 = plt.figure(7)
    plt.subplot(2,1,1)
    plt.imshow(cluster_moving_neuron)
    #plt.colorbar()
    plt.title('Moving_102246_Immobile Cluster')
    plt.xlabel('Time (units?)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,1,2)
    plt.imshow(cluster_immobile_neuron)
    #plt.colorbar()
    plt.title('Immobile_102246_Immobile Cluster')
    plt.xlabel('Time (units?)')
    plt.ylabel('Sorted Neuron')

    plot8 = plt.figure(8)
    plt.subplot(2,1, 1)
    plt.imshow(moving_cluster_moving_neuron)
    #plt.colorbar()
    plt.title('Moving_102246_Moving Cluster')
    plt.xlabel('Time (units?)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,1, 2)
    plt.imshow(moving_cluster_immobile_neuron)
    #plt.colorbar()
    plt.title('Immobile_102246_Moving Cluster')
    plt.xlabel('Time (units?)')
    plt.ylabel('Sorted Neuron')


    plt.show()






