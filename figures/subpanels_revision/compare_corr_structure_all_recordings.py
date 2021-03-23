"""
Created 10 Dec 2020
We had obseved in our moving-immobile tranistino experiment that the correlation structure changes dramaticallhy
And we would like to know how much correlation structure normally changes in a  moving worm.
So this script goes through each recording, splits it into two and then compares the two correlation structures.
To give us a baseline for comparison.
leifer@princeton.edu
"""

################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np
from numpy.core._multiarray_umath import ndarray

from prediction import userTracker
import prediction.dataHandler as dh


def dissimilarity(A, B):
    # calculate the dissimilarity between two matrices
    # using the RMS difference of paired correlation coefficients
    N = A.shape[0]
    return np.sqrt(np.mean(np.square(A - B))) / np.sqrt(N * (N - 1))


def main():

    codePath = userTracker.codePath()
    outputFolder = os.path.join(codePath,'figures/subpanels_revision/generatedFigs')

    data = {}
    for typ in ['AKS297.51', 'AML32', 'AML18']:
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





    ### Plot Heatmap for each recording





    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import matplotlib.pylab as pylab



    dissim={}
    for key in ['AKS297.51_moving', 'AML32_moving']:#,  'AML18_moving']:
        dissim[key]=[]
        dset = data[key]['input']
        # For each recording
        for idn in dset.keys():
            activity = dset[idn]['Neurons']['I_smooth_interp_crop_noncontig']
            rec_length = activity.shape[1]
            halfway = np.floor_divide(rec_length, 2)
            first, second = np.int(0.30 * rec_length), np.int(0.40 * rec_length)
            cmat_first, cmat_second = np.corrcoef(activity[:,:first]), np.corrcoef(activity[:,second:])
            dissim[key].append(dissimilarity(cmat_first, cmat_second))

    # From CheckNeuralTrajectories for 193044  and BrainScanner20180511_134913 (for the latter we probably have to rerun w/ a gap betwen mov and immobile)
    #And also for BrainScanner20200915_144610 from TransitionAnalysis2020_second.py
    transition = np.array([0.0056080890227034885, 0.004666316998854658, 0.005891564672126974])# 0.005336660925126822])

    moving_gcamp = np.concatenate((np.array(dissim['AKS297.51_moving']), np.array(dissim['AML32_moving'])))
    #moving_gfp = np.array(dissim['AML18_moving'])  # type: ndarray
    combined_data = [moving_gcamp,  transition]
    from scipy import stats
    t, p_gcamp = stats.ttest_ind(moving_gcamp, transition, equal_var=False)

    labels = ["Moving to Moving", "Moving to Immobile", "Moving to Movin GFP"]

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    sns.set_style("white")
    sns.set_style("ticks")
    ax = sns.swarmplot(data=combined_data, color="black")
    ax = sns.boxplot(data=combined_data,
                     showcaps=False, boxprops={'facecolor': 'None'},
                     showfliers=False, whiskerprops={'linewidth': 0})
    ax.set_ylim(0, .7*ax.get_ylim()[1])
    ax.set_xticklabels(labels)
    yloc = plt.MaxNLocator(5)
    ax.yaxis.set_major_locator(yloc)
    sns.despine()
    ax.set_title('welch unequal t-test: %0.4f' % p_gcamp)
    ax.set_ylabel('Dissimilarity, ||A-B||/N')
    plt.show()
    print("plotted")




if __name__ == '__main__':
    print("about to run main()")
    main()