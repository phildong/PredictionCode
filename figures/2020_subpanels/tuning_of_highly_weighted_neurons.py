import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
from prediction import userTracker
from prediction import dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
import prediction.provenance as prov

behavior = 'velocity'
pickled_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_' + behavior + '_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

def take_deriv(neurons):
    nan_zero = np.copy(neurons)
    nan_zero[np.isnan(neurons)] = 0
    nan_zero_filtered = gaussian_filter(nan_zero, order = 1, sigma = (0, 7))
    flat = 0*neurons.copy()+1
    flat[np.isnan(neurons)] = 0
    flat_filtered = gaussian_filter(flat, order = 0, sigma = (0, 7))
    deriv = nan_zero_filtered/flat_filtered
    return deriv

neuron_data = {}
deriv_neuron_data = {}
time_data = {}
vel_data = {}
for typ_cond in ['AKS297.51_moving']:
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

    for key in keyList:
        if key in excludeSets:
            continue
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        vel = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']


        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
                vel = vel[idxs]

        neuron_data[key] = neurons
        deriv_neuron_data[key] = take_deriv(neurons)
        time_data[key] = time
        vel_data[key] = vel


key='BrainScanner20200130_110803'

import os
outfilename = key + '_highweight_tuning.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), outfilename))

#Sort neurons by abs value weight
slm_weights_raw = data[key]['slm_with_derivs']['weights'][:data[key]['slm_with_derivs']['weights'].size / 2]
slm_weights_raw_deriv = data[key]['slm_with_derivs']['weights'][data[key]['slm_with_derivs']['weights'].size / 2:]
highly_weighted_neurons = np.flipud(np.argsort(np.abs(slm_weights_raw)))
highly_weighted_neurons_deriv = np.flipud(np.argsort(np.abs(slm_weights_raw_deriv)))
num_neurons=len(highly_weighted_neurons)

for rank in np.arange(num_neurons):
    neuron = highly_weighted_neurons[rank]
    weight= slm_weights_raw[neuron]
    activity = neuron_data[key][neuron]

    #Calculate bins for box plot and split data up into subarrays based on bin
    nbins = 10
    plus_epsilon = 1.00001
    bin_edges = np.linspace(np.nanmin(vel_data[key]) * plus_epsilon, np.nanmax(vel_data[key]) * plus_epsilon, nbins)
    binwidth = np.diff(bin_edges)
    assigned_bin = np.digitize(vel_data[key], bin_edges)
    activity_bin = [None] * (len(bin_edges) - 1)  # note the activity has to be lists, and there should be 1 less because the bins are edges
    for k, each in enumerate(np.unique(assigned_bin)):
        activity_bin[k] = activity[np.argwhere(assigned_bin == each)[:, 0]]

    fig1 = plt.figure(constrained_layout=True, figsize=[15, 8])
    fig1.suptitle(key + ' Neuron: %d, Weight Rank: %d, Weight = %.4f' % (neuron, rank, weight))
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig1)

    #Generate box plot
    f1_ax1 = fig1.add_subplot(gs[0, 0], xlabel=behavior, ylabel='Activity (F)')
    f1_ax1.plot(vel_data[key], activity, 'o', alpha=.05)
    boxprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=3.5)
    labels = [''] * len(activity_bin)
    f1_ax1.boxplot(activity_bin, positions=bin_edges[:-1] + binwidth / 2, widths=binwidth * .9, boxprops=boxprops,
                medianprops=medianprops, labels=labels, manage_xticks=False)
    plt.locator_params(nbins=4)

    f1_ax2 = fig1.add_subplot(gs[0,1:], xlabel='time (s)', ylabel='Activity')
    f1_ax2.plot(time_data[key], activity)
    f1_ax3 = fig1.add_subplot(gs[1,1:], xlabel='time (s)', ylabel=behavior)
    f1_ax3.plot(time_data[key], data[key]['slm_with_derivs']['output'], 'k')
    prov.stamp(f1_ax3, .55, .35, __file__ + '\n' + pickled_data)
    pdf.savefig(fig1)

pdf.close()
print("wrote " + outfilename)

