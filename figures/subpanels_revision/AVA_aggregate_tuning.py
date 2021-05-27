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

DERIV = True

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

    if DERIV:
        from prediction.Classifier import rectified_derivative
        _, _, activity = rectified_derivative(dset['Neurons']['I_smooth_interp_crop_noncontig'])
        color = u'#ff7f0e'
        type = 'dF/dt'
    else:
        activity = dset['Neurons']['I_smooth_interp_crop_noncontig']
        color = u'#1f77b4'
        type = 'F'

    time = dset['Neurons']['I_Time_crop_noncontig']

    numNeurons = activity.shape[0]
    comVel = dset['Behavior_crop_noncontig']['CMSVelocity']
    vel = comVel
    curv = dset['Behavior_crop_noncontig']['Curvature']
    psvel = dset['Behavior_crop_noncontig']['PhaseShiftVelocity']

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

# Calculate bins for box plot and split data up into subarrays based on bin
nbins = 13
plus_epsilon = 1.00001
bin_edges = np.linspace(np.nanmin(vel_bucket) * plus_epsilon, np.nanmax(vel_bucket) * plus_epsilon, nbins)
binwidth = np.diff(bin_edges)
assigned_bin = np.digitize(vel_bucket, bin_edges)
activity_bin = [None] * (len(
    bin_edges) - 1)  # note the activity has to be lists, and there should be 1 less because the bins are edges
for k, each in enumerate(np.unique(assigned_bin)):
    activity_bin[k] = z_activity[np.argwhere(assigned_bin == each)[:, 0]]


plt.figure(figsize=[4,3.6]) #width, height
#plt.axhline(dashes=[3, 3], lw=0.5, color="black", zorder=0)
#plt.axvline(dashes=[3, 3], lw=0.5, color="black", zorder=1)
plt.plot(vel_bucket, z_activity, '.', color=color, alpha=.05, zorder=10)
if False:
    import numpy.polynomial.polynomial as poly
    try:
        coefs = poly.polyfit(vel_bucket, z_activity, 1)
        x_new = np.linspace(np.min(vel_bucket), 0, num=10)
        ffit = poly.polyval(x_new, coefs)
        plt.plot(x_new, ffit, 'r--', zorder=9)
    except:
        None
boxprops = dict(linewidth=.5)
capprops = dict(linewidth=.5)
whiskerprops = dict(linewidth=.5)
flierprops = dict(linewidth=.2, markersize=1, marker='+')
medianprops = dict(linewidth=2, color='k')#'#67eb34')
labels = [''] * len(activity_bin)
plt.boxplot(activity_bin, positions=bin_edges[:-1] + binwidth / 2, widths=binwidth * .9, boxprops=boxprops,
               medianprops=medianprops, labels=labels, manage_xticks=False,
               capprops=capprops, whiskerprops=whiskerprops, flierprops=flierprops, zorder=20)
plt.locator_params(nbins=5)
plt.axhline(color='k', linewidth=.5)
plt.xlim([-.2, 0.3])
plt.xlabel('velocity (mm^-1 s)')
plt.ylabel('AVA activity (z-score(' + type + '))')
plt.legend()
plt.show()
print("done")

