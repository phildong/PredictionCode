# Script to read in neuron data and do correlations
# First do correlation for all neuronxall neuron
# Using that matrix of data, pull out the data of AVA specific correlations

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16

# For data set 102246 (moving-immobile)-Moving 1-1590, Immobile 1700-4016 (maybe less/still immobile then), AVA 29 (30 matlab) (and 17?)
# For data set 111126 (moving-immobile)-moving 1-1785, Immobile 1900-3245, AVA 27 (28 matlab)
# For data set 144610 (moving-immobile)-moving 0-1429, Immobile 1539-2904, BFP 2924-3nd, AVA- 17 and 81

#For Dataset 173358-immobile only curre

import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap


def get_data(start_frame, end_frame, neurons, neurons_withNaN, neurons_Zscore, time, time_contig, velocity):
    # take the loaded in data and pull out the neuron data for moving or immobile.
    # look up time of start frame/end frame in the time_contig
    # see what index in time (noncontig) is closest to time_contig value
    start_time= time_contig[start_frame]
    start_index = np.abs(time - start_time).argmin()
    end_time = time_contig[end_frame]
    end_index = np.abs(time-end_time).argmin()
    neuron_data = neurons[:, start_index:end_index]
    neuron_NaN_data = neurons_withNaN[:, start_index:end_index]
    zscore_neuron = neurons_Zscore[:, start_index:end_index]
    [row, column] = neurons.shape
    neuron_num = list(range(0, row, 1))
    crop_velocity = velocity[start_frame:end_frame]
    crop_time = time_contig[start_frame:end_frame]
    return neuron_data, neuron_NaN_data, zscore_neuron, neuron_num, crop_velocity, crop_time


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


def insert_Nan_neurons(neuron_list, index_list):
    # replace the untracked neurons immobile data with NaN in the moving data
    for i in index_list:
        neuron_list[i, :] = np.NaN
    return neuron_list


def do_residual(moving_worm, immobile_worm):
    # this function should take the residual of whatever we give it (should work for whole matrix correlation as well
    # as AVA specific correlation)
    residual = moving_worm- immobile_worm
    return residual


def do_clustering(matrix):
    cg = clustermap(matrix)
    cgIdx = np.array(cg.data2d.columns)
    #cgIdx is a list of the resorted neuron numbers. Want to mark neurons for AVA
    return cgIdx

def get_index_AVA_clustered(AVAneuron, cgIdx):
    #take the cgIdx and find what index corresponds to the AVA neuron
    AVA_index = np.where(cgIdx == AVAneuron)
    return AVA_index

def cluster_matrix(matrix, cgIdx):
    clustered_1 = matrix[:, cgIdx]
    clustered_matrix = clustered_1[cgIdx, :]
    return clustered_matrix


def remove_nan(matrix, index_list):
    # remove the matrix rows/columns that are the Nan values
    # remove the nan_rows
    matrix_no_row_nan = np.delete(matrix, index_list,0)
    # remove the nan_columns
    matrix_no_nan = np.delete(matrix_no_row_nan, index_list, 1)
    return matrix_no_nan


def remove_and_cluster_and_residual(matrix_move, matrix_immobile, index_list):
    matrix_move_noNaN = remove_nan(matrix_move, index_list)
    matrix_immobile_noNaN = remove_nan(matrix_immobile, index_list)
    #do immobile clustering
    immobile_cgIdx = do_clustering(matrix_immobile_noNaN)
    moving_cgIdx = do_clustering(matrix_move_noNaN)
    movecluster_AVA_index1 = get_index_AVA_clustered(17, moving_cgIdx)
    movecluster_AVA_index2 = get_index_AVA_clustered(81,moving_cgIdx)
    movecluster_AVA = [movecluster_AVA_index1, movecluster_AVA_index2]
    immcluster_AVA_index1 = get_index_AVA_clustered(17, immobile_cgIdx)
    immcluster_AVA_index2 = get_index_AVA_clustered(81,immobile_cgIdx)
    immcluster_AVA = [immcluster_AVA_index1, immcluster_AVA_index2]
    movecluster_move_noNaN = cluster_matrix(matrix_move_noNaN, moving_cgIdx)
    movecluster_immobile_noNaN = cluster_matrix(matrix_immobile_noNaN, moving_cgIdx)
    immcluster_move_noNaN = cluster_matrix(matrix_move_noNaN, immobile_cgIdx)
    immcluster_immobile_noNaN = cluster_matrix(matrix_immobile_noNaN, immobile_cgIdx)
    movecluster_residual = do_residual(movecluster_move_noNaN, movecluster_immobile_noNaN)
    immcluster_residual = do_residual(immcluster_move_noNaN, immcluster_immobile_noNaN)
    return movecluster_move_noNaN, movecluster_immobile_noNaN, movecluster_residual, movecluster_AVA, immcluster_move_noNaN, immcluster_immobile_noNaN, immcluster_residual, immcluster_AVA

def sum_matrix(matrix):
    # sum together all of the elements of the matrix. subtract the 1:1 correlation, divide by matrix
    #take abs value of matrix
    abs_matrix = np.abs(matrix)
    total = abs_matrix.sum()
    diagonal = len(matrix)
    corr = total-diagonal # subtract off the 1:1
    elements = (diagonal-1)*diagonal
    avg_corr = corr/elements
    return avg_corr

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
    for key in filter(lambda x: '144610' in x, keyList):
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

    moving_data, moving_withNaN, moving_zscore, neuron_number, moving_velocity, moving_time = get_data(0, 1429, neurons, neurons_withNaN, neurons_ZScore, time, time_contig,velocity)
    moving_corr, moving_r_square = do_correlation_all(moving_data)
    AVA_corr_noNan1, AVA_r_square_noNan1 = do_correlation_AVA(moving_corr,moving_r_square, 17)
    AVA_corr_noNan2, AVA_r_square_noNan2 = do_correlation_AVA(moving_corr,moving_r_square, 81)


    # cgIdx = do_clustering(moving_corr_NaN) cluster based on immobile

    immobile_data, data_withNaN, immobile_zscore, neuron_number_im, imm_velocity, imm_time = get_data(1539, 2904, neurons, neurons_withNaN, neurons_ZScore, time, time_contig, velocity)
    immobile_corr, immobile_r_square = do_correlation_all(immobile_data)
    AVA_corr_im1, AVA_r_square_im1 = do_correlation_AVA(immobile_corr, immobile_r_square, 17)
    AVA_corr_im2, AVA_r_square_im2 = do_correlation_AVA(immobile_corr, immobile_r_square, 81)
    nan_index = find_NaN_neurons(data_withNaN, neuron_number)
    immobile_corr_NaN = replace_NaN_neurons(immobile_corr, nan_index, neuron_number) # Nan = 0
    immobile_r_NaN = replace_NaN_neurons(immobile_r_square, nan_index, neuron_number)


    moving_corr_NaN = replace_NaN_neurons(moving_corr, nan_index, neuron_number)
    moving_r_square_NaN = replace_NaN_neurons(moving_r_square, nan_index, neuron_number)

    #Remove Nan, cluster and do residuals on correlation matrix
    mcluster_move_noNaN, mcluster_imm_noNaN, mcluster_residual, mcluster_AVA, imcluster_move_noNaN, imcluster_imm_noNaN, imcluster_residual, imcluster_AVA = remove_and_cluster_and_residual(moving_corr, immobile_corr, nan_index)
    move_noNaN_avg_corr = sum_matrix(mcluster_move_noNaN)
    imm_nonNaN_avg_corr = sum_matrix(mcluster_imm_noNaN)
    move_noNaN_avg_corr = round(move_noNaN_avg_corr,2)
    imm_nonNaN_avg_corr = round(imm_nonNaN_avg_corr,2)


    #Put NaN/0s into the AVA measurements as well, propagating them to the moving part
    AVA_corr1 = replace_Nan_AVA(AVA_corr_noNan1, nan_index)
    AVA_r_square1 = replace_Nan_AVA(AVA_r_square_noNan1, nan_index)

    AVA_corr2 = replace_Nan_AVA(AVA_corr_noNan2, nan_index)
    AVA_r_square2 = replace_Nan_AVA(AVA_r_square_noNan2, nan_index)



    cgIdx_immobile = do_clustering(immobile_corr_NaN)
    clustered_immobile_corr = cluster_matrix(immobile_corr_NaN, cgIdx_immobile)
    clustered_immobile_r = cluster_matrix(immobile_r_NaN, cgIdx_immobile)
    AVA_index_imm1 = get_index_AVA_clustered(17, cgIdx_immobile)
    AVA_index_imm2 = get_index_AVA_clustered(81, cgIdx_immobile)

    clustered_moving_corr = cluster_matrix(moving_corr_NaN, cgIdx_immobile)
    clustered_moving_r = cluster_matrix(moving_r_square_NaN, cgIdx_immobile)


    # cluster the neuron signal
    data_withNaN = insert_Nan_neurons(data_withNaN, nan_index)
    moving_withNaN = insert_Nan_neurons(moving_withNaN, nan_index)
    immobile_zscore = insert_Nan_neurons(immobile_zscore, nan_index)
    moving_zscore = insert_Nan_neurons(moving_zscore, nan_index)
    cluster_immobile_neuron = cluster_calcium(data_withNaN, cgIdx_immobile)
    cluster_moving_neuron = cluster_calcium(moving_withNaN, cgIdx_immobile)
    cluster_immobile_zscore = cluster_calcium(immobile_zscore, cgIdx_immobile)
    cluster_moving_zscore = cluster_calcium(moving_zscore, cgIdx_immobile)


    # calculate residuals
    corr_residual = do_residual(clustered_moving_corr, clustered_immobile_corr)
    r_square_residual = do_residual(clustered_moving_r, clustered_immobile_r)
    ava_corr_residual1 = do_residual(AVA_corr1, AVA_corr_im1)
    ava_corr_residual2 = do_residual(AVA_corr2, AVA_corr_im2)
    ava_r_residual1 = do_residual(AVA_r_square1, AVA_r_square_im1)
    ava_r_residual2 = do_residual(AVA_r_square2, AVA_r_square_im2)




    # do the clustering on moving data
    cgIdx_moving = do_clustering(moving_corr_NaN)
    moving_cluster_moving_corr = cluster_matrix(moving_corr_NaN, cgIdx_moving)
    moving_cluster_immobile_corr = cluster_matrix(immobile_corr_NaN, cgIdx_moving)
    moving_cluster_moving_r = cluster_matrix(moving_r_square_NaN, cgIdx_moving)
    moving_cluster_immobile_r = cluster_matrix(immobile_r_NaN, cgIdx_moving)
    AVA_index_mov1 = get_index_AVA_clustered(17, cgIdx_moving)
    AVA_index_mov2 = get_index_AVA_clustered(81, cgIdx_moving)

    moving_corr_residual = do_residual(moving_cluster_moving_corr, moving_cluster_immobile_corr)
    moving_r_residual = do_residual(moving_cluster_moving_r, moving_cluster_immobile_r)

    moving_cluster_immobile_neuron = cluster_calcium(data_withNaN, cgIdx_moving)
    moving_cluster_moving_neuron = cluster_calcium(moving_withNaN, cgIdx_moving)
    moving_cluster_immobile_zscore = cluster_calcium(immobile_zscore, cgIdx_moving)
    moving_cluster_moving_zscore = cluster_calcium(moving_zscore, cgIdx_moving)

    AVA_label = [17, 81] #For dataset 144610
   #PLOTTING

    plot1 = plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(clustered_moving_corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_144610_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(clustered_immobile_corr,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_144610_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(corr_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plot2 = plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(clustered_moving_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R^2_144610_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(clustered_immobile_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R^2_144610_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(r_square_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_14461')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_imm1,AVA_index_imm2], AVA_label)
    plt.yticks([AVA_index_imm1, AVA_index_imm2], AVA_label)

    plot3 = plt.figure(3)
    plt.subplot(2,3,1)
    plt.bar(neuron_number, AVA_corr1)
    plt.title('AVA Correlation 17_144610_moving')
    plt.xlabel('Neuron')
    plt.ylabel('Correlation')
    plt.ylim(-0.8, 1)
    plt.grid()

    plt.subplot(2,3,2)
    plt.bar(neuron_number, AVA_corr_im1)
    plt.title('AVA Correlation 17_144610_immobile')
    plt.xlabel('Neuron')
    plt.ylabel('Correlation')
    plt.ylim(-0.8, 1)
    plt.grid()

    plt.subplot(2,3,3)
    plt.bar(neuron_number, ava_corr_residual1)
    plt.title('Residual_moving-immobile_1144610')
    plt.xlabel('Neuron')
    plt.ylabel('Residual')
    #plt.ylim(-0.6, 1)
    plt.grid()

    plt.subplot(2,3,4)
    plt.bar(neuron_number, AVA_corr2)
    plt.title('AVA Correlation 81_144610_moving')
    plt.xlabel('Neuron')
    plt.ylabel('Correlation')
    plt.ylim(-0.8, 1)
    plt.grid()

    plt.subplot(2,3,5)
    plt.bar(neuron_number, AVA_corr_im2)
    plt.title('AVA Correlation 81_144610_immobile')
    plt.xlabel('Neuron')
    plt.ylabel('Correlation')
    plt.ylim(-0.8, 1)
    plt.grid()

    plt.subplot(2,3,6)
    plt.bar(neuron_number, ava_corr_residual2)
    plt.title('Residual_moving-immobile_1144610')
    plt.xlabel('Neuron')
    plt.ylabel('Residual')
    #plt.ylim(-0.6, 1)
    plt.grid()

    plot4 = plt.figure(4)
    plt.subplot(2,3,1)
    plt.bar(neuron_number, AVA_r_square1)
    plt.title('AVA R^2 17_144610_moving')
    plt.xlabel('Neuron')
    plt.ylabel('R^2')
    plt.ylim(0, 1)
    plt.grid()

    plt.subplot(2,3,2)
    plt.bar(neuron_number, AVA_r_square_im1)
    plt.title('AVA R^2 17_144610_immobile')
    plt.xlabel('Neuron')
    plt.ylabel('R^2')
    plt.ylim(0, 1)
    plt.grid()

    plt.subplot(2,3,3)
    plt.bar(neuron_number, ava_r_residual1)
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Neuron')
    plt.ylabel('Residual')
    #plt.ylim(0, 1)
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.bar(neuron_number, AVA_r_square2)
    plt.title('AVA R^2 81_144610_moving')
    plt.xlabel('Neuron')
    plt.ylabel('R^2')
    plt.ylim(0, 1)
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.bar(neuron_number, AVA_r_square_im2)
    plt.title('AVA R^2 81_144610_immobile')
    plt.xlabel('Neuron')
    plt.ylabel('R^2')
    plt.ylim(0, 1)
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.bar(neuron_number, ava_r_residual2)
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Neuron')
    plt.ylabel('Residual')
    # plt.ylim(0, 1)
    plt.grid()
    #######


    plot5 = plt.figure(5)
    plt.subplot(1,3,1)
    plt.imshow(moving_cluster_moving_corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_144610_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(moving_cluster_immobile_corr,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation_144610_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(moving_corr_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)


    plot6 = plt.figure(6)
    plt.subplot(1,3,1)
    plt.imshow(moving_cluster_moving_r, vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R_Square_144610_moving')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(moving_cluster_immobile_r,vmin=0, vmax=1)
    plt.colorbar()
    plt.title('R_Square_144610_immobile')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(moving_r_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')
    plt.xticks([AVA_index_mov1,AVA_index_mov2], AVA_label)
    plt.yticks([AVA_index_mov1, AVA_index_mov2], AVA_label)

    plot7 = plt.figure(7)
    plt.subplot(2,2,1)
    plt.imshow(cluster_moving_neuron)
    #plt.colorbar()
    plt.title('Moving_144610_Immobile Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,2,2)
    plt.imshow(cluster_immobile_neuron)
    #plt.colorbar()
    plt.title('Immobile_144610_Immobile Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,2,3)
    plt.imshow(cluster_moving_zscore)
    #plt.colorbar()
    plt.title('Mov_ZScore_144610_Immobile Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,2,4)
    plt.imshow(cluster_immobile_zscore)
    #plt.colorbar()
    plt.title('Imm_Zscore_144610_Immobile Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plot8 = plt.figure(8)
    plt.subplot(2,2, 1)
    plt.imshow(moving_cluster_moving_neuron)
    #plt.colorbar()
    plt.title('Moving_144610_Moving Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,2, 2)
    plt.imshow(moving_cluster_immobile_neuron)
    #plt.colorbar()
    plt.title('Immobile_144610_Moving Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2,2, 3)
    plt.imshow(moving_cluster_moving_zscore)
    #plt.colorbar()
    plt.title('Mov_ZScore_144610_Moving Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.subplot(2, 2,4)
    plt.imshow(moving_cluster_immobile_zscore)
    #plt.colorbar()
    plt.title('Imm_ZScore_144610_Moving Cluster')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plot9 = plt.figure(9)
    plt.subplot(1,3,1)
    plt.imshow(mcluster_move_noNaN, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Move_144610_AvgCorr ='+str(move_noNaN_avg_corr))
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(mcluster_AVA, AVA_label)
    plt.yticks(mcluster_AVA, AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(mcluster_imm_noNaN,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Imm_144610_AvgCorr =' +str(imm_nonNaN_avg_corr))
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(mcluster_AVA, AVA_label)
    plt.yticks(mcluster_AVA, AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(mcluster_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(mcluster_AVA, AVA_label)
    plt.yticks(mcluster_AVA, AVA_label)

    plot10 = plt.figure(10)
    plt.subplot(1,3,1)
    plt.imshow(imcluster_move_noNaN, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Move_144610_AvgCorr ='+str(move_noNaN_avg_corr))
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(imcluster_AVA, AVA_label)
    plt.yticks(imcluster_AVA, AVA_label)

    plt.subplot(1,3,2)
    plt.imshow(imcluster_imm_noNaN,vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Imm_144610_AvgCorr =' +str(imm_nonNaN_avg_corr))
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(imcluster_AVA, AVA_label)
    plt.yticks(imcluster_AVA, AVA_label)

    plt.subplot(1,3,3)
    plt.imshow(imcluster_residual, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Residual_moving-immobile_144610')
    plt.xlabel('Sorted Neuron_NaN removed')
    plt.ylabel('Sorted Neuron_NaN removed')
    plt.xticks(imcluster_AVA, AVA_label)
    plt.yticks(imcluster_AVA, AVA_label)

    plot11 = plt.figure(11)
    plt.subplot(3,1,1)
    plt.plot(moving_velocity) #, moving_time)
    plt.title('Velocity')
    plt.xlabel('time_contig')

    plt.subplot(3,1,2)
    plt.plot(moving_withNaN[17,:]) #, moving_time)
    plt.title('AVA Neuron 17')
    plt.xlabel('time_contig')

    plt.subplot(3,1,3)
    plt.plot(moving_withNaN[81,:])#, moving_time)
    plt.title('AVA Neuron 81')
    plt.xlabel('time_contig')
    plt.show()
