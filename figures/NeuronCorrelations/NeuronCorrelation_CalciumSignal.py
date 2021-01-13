
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
# For dataset 122300 (immobile part only analyze)- immobile 1600-2910, BFP 2924-end, AVA 94 and 23?


def get_data(start_frame, end_frame, neurons, neurons_withNaN, neurons_Zscore, time, time_contig):
    # take the loaded in data and pull out the neuron data for moving or immobile.
    # look up time of start frame/end frame in the time_contig
    # see what index in time (noncontig) is closest to time_contig value
    start_time= time_contig[start_frame]
    start_index = np.abs(time - start_time).argmin()
    end_time = time_contig[end_frame]
    end_index = np.abs(time-end_time).argmin()
    neuron_data = neurons[:, start_index:end_index]


    neuron_NaN_data = neurons_withNaN[:, :640]
    zscore_neuron = neurons_Zscore[:, :640]
    [row, column] = neurons.shape
    neuron_num = list(range(0, row, 1))
    return neuron_data, neuron_NaN_data, zscore_neuron, neuron_num


def do_correlation_all(neuron_data):
    # take in the Ratio2 frames, do the correlations/make the matrix
    r = np.corrcoef(neuron_data)
    # square the matrix to get the r_square
    r_square = np.square(r)
    return r, r_square


def do_clustering(matrix):
    cg = clustermap(matrix)
    cgIdx = np.array(cg.data2d.columns)
    return cgIdx


def cluster_matrix(matrix, cgIdx):
    clustered_1 = matrix[:, cgIdx]
    clustered_matrix = clustered_1[cgIdx, :]
    return clustered_matrix

def cluster_calcium(neurons_NaN, cgIdx):
    cluster_neuron = neurons_NaN[cgIdx, :]
    return cluster_neuron


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

def remove_nan(matrix):
    # remove the matrix rows/columns that are the Nan values
    # remove the nan_rows
    index_list = filter(lambda x:np.nonzero(matrix[x,:])[0].size ==0,range(matrix.shape[0]))
    matrix_no_row_nan = np.delete(matrix, index_list,0)
    # remove the nan_columns
    matrix_no_nan = np.delete(matrix_no_row_nan, index_list, 1)
    return matrix_no_nan


def replace_NaN_neurons(r, neuron_num):
    # for the moving-immobile transition, use index list to see where non tracked neurons are. Replace those
    # rows/columns with 0s in the r/r_square
    index_list = filter(lambda x:np.all(np.isnan(r[x,:])),range(r.shape[0]))
    for i in index_list:
        # each index in the list is a row of NaNs. replace the correlation with 0s if neuron isn't tracked
        r[i, :] = np.zeros(len(neuron_num))
        r[:, i] = np.zeros(len(neuron_num))
    return r


for typ_cond in ['AML310_moving']: #, 'AML310_moving']:
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
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate = folder, dataPars = dataPars)
    keyList = np.sort(dataSets.keys())
    for key in filter(lambda x: '110803' in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_withNaN = dataSets[key]['Neurons']['I_smooth'] # use this to find the untracked neurons after transition
        neurons_ZScore = dataSets[key]['Neurons']['ActivityFull'] # Z scored neurons to use to look at calcium traces
        # velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        # curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
        # For immobile- how is NaN neurons that are not hand tracked dealt with by the smooth_interp...
        # Still do the correlation with all (the interpolated ones too, but then replace with 0s)?

    data, immobile_withNaN, immobile_zscore, neuron_number = get_data(1,1465, neurons, neurons_withNaN, neurons_ZScore, time, time_contig)
    immobile_corr, immobile_r_square = do_correlation_all(data)
    #index = find_NaN_neurons(immobile_withNaN, neuron_number)

    immobile_corr_NaN = replace_NaN_neurons(immobile_corr, neuron_number)

    cgIdx_immobile = do_clustering(immobile_corr_NaN)
    clustered_immobile_corr = cluster_matrix(immobile_corr_NaN, cgIdx_immobile)  # Why can't I do this in one line? or is there a better way to do this?

    cluster_neuron = cluster_calcium(data, cgIdx_immobile)
    cluster_neuron_zscore = (cluster_neuron-np.mean(cluster_neuron, axis = 1)[:,np.newaxis])/np.std(cluster_neuron, axis =1)[:,np.newaxis]
    index_list = filter(lambda x:np.all(np.isnan(cluster_neuron_zscore[x,:])),range(cluster_neuron_zscore.shape[0]))
    cluster_neuron_zscore_nonan = np.delete(cluster_neuron_zscore, index_list,0)

    cluster_imm_corr_noNan = remove_nan(clustered_immobile_corr)


    plot3 = plt.figure(3)

    #plt.imshow(immobile_corr)
    plt.imshow(cluster_imm_corr_noNan, vmin=-1, vmax = 1)
    plt.colorbar()
    plt.title('Correlation_110803')
    plt.xlabel('Sorted Neuron')
    plt.ylabel('Sorted Neuron')

    plot2 = plt.figure(2)
    plt.imshow(cluster_neuron_zscore_nonan, vmin = -3, vmax = 3)
    plt.colorbar()
    plt.title('Calcium Signal_110803')
    plt.xlabel('Time (sec/6)')
    plt.ylabel('Sorted Neuron')

    plt.show()

