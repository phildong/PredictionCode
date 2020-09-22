################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh

print("Loading data..")

codePath = userTracker.codePath()
outputFolder = os.path.join(codePath,'figures/Debugging')

data = {}
for typ in ['AKS297.51']:
    for condition in ['moving']:  # ['moving', 'immobilized', 'chip']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path, '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path, 'Analysis/{}_{}.hdf5'.format(typ, condition))

        print("Loading " + folder + "...")
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
print('Done reading data.')



### CHOOSE DATASET TO PLOT
keys = ['AKS297.51_moving',            'AKS297.51_moving',             'AKS297.51_moving',          'AKS297.51_moving',             'AKS297.51_moving',           'AKS297.51_moving',            'AKS297.51_moving' ]
idns = ['BrainScanner20200130_110803', 'BrainScanner20200130_110803', 'BrainScanner20200130_105254', 'BrainScanner20200310_141211', 'BrainScanner20200310_141211', 'BrainScanner20200310_142022', 'BrainScanner20200310_142022'       ]
neurons = [32,                          15,                           95,                             71,                           42,                            15,                                     16        ]

z_activity = np.array([])
vel_bucket = np.array([])
import itertools
for  key, idn, neuron in itertools.izip(keys, idns, neurons):
    print(key, idn, neuron)

    ### Get the relevant data.
    dset = data[key]['input'][idn]
    activity = dset['Neurons']['I_smooth_interp_crop_noncontig']
    time = dset['Neurons']['I_Time_crop_noncontig']

    numNeurons = activity.shape[0]
    vel = dset['Behavior_crop_noncontig']['AngleVelocity']
    comVel = dset['Behavior_crop_noncontig']['CMSVelocity']
    curv = dset['Behavior_crop_noncontig']['Eigenworm3']

    from scipy import stats
    z_activity = np.append(z_activity, stats.zscore(activity[neuron]))
    vel_bucket = np.append(vel_bucket, vel)


    SHOW_PER_NEURON_DETAILS = False
    if SHOW_PER_NEURON_DETAILS:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(vel, activity[neuron], 'o');
        plt.xlabel('Eigen Velocity')
        plt.ylabel('Calcium Activity')
        plt.title(idn + 'Neuron ' + str(neuron))

        plt.figure()
        plt.plot(time, activity[neuron], label="activity");
        plt.plot(time, vel*100, label="vel*100  ");
        plt.xlabel('Time')
        plt.legend()
        plt.title(idn + 'Neuron ' + str(neuron))

        plt.figure()
        plt.plot(comVel, activity[neuron], 'o');
        plt.xlabel('CoM Vel')
        plt.ylabel('Calcium Activity')
        plt.title(idn + 'Neuron ' + str(neuron))

        plt.figure()
        plt.plot(comVel*2000, label="comVel*2000")
        plt.plot(vel, label="eigenworm velocity")
        plt.legend()
        plt.title('velocity')

print("plotting aggregate figure")
import matplotlib.pyplot as plt
#import seaborn as sns
#plt.style.use('seaborn')
nbins=10
bin_means, bin_edges, binnumber = stats.binned_statistic(vel_bucket,
                z_activity, statistic='median', bins=nbins)
plt.figure()
plt.axhline(dashes=[3, 3], lw=0.5, color="black", zorder=0)
plt.axvline(dashes=[3, 3], lw=0.5, color="black", zorder=1)
plt.plot(vel_bucket, z_activity, '.', color='gray', label='raw data', alpha=.07, zorder=10)
plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='orange', lw=5,
           label='binned median of data', alpha=1, zorder=20)
plt.xlabel('Body bend velocity (rad/s)')
plt.ylabel('AVA activity (z-score(F))')
plt.legend()
plt.show()
print("done")

