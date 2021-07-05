import user_tracker
import data_handler as dh

import numpy as np
from scipy.ndimage import gaussian_filter1d

import os
import pickle

def dFdt(neurons):
    """Returns a matrix for which each row is a dF/dt signal, computed using a Gaussian derivative kernel of width 2*7/6 = 2.3 s"""
    nan_zero = np.copy(neurons)
    nan_zero[np.isnan(neurons)] = 0
    nan_zero_filtered = gaussian_filter1d(nan_zero, 7, order = 1)

    flat = 0*neurons.copy()+1
    flat[np.isnan(neurons)] = 0
    flat_filtered = gaussian_filter1d(flat, 7, order = 0)

    with np.errstate(invalid = 'ignore'):
        return nan_zero_filtered/flat_filtered

# Intervals where tracking is lost and data is unreliable
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}


for gtype in ['gcamp', 'gfp']:
    data = {}
    for typ_cond in (['AKS297.51_moving', 'AML32_moving'] if gtype == 'gcamp' else ['AML18_moving']):
        path = user_tracker.dataPath()
        folder = os.path.join(path, '%s/' % typ_cond)
        dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))
        
        dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
        keyList = np.sort(dataSets.keys())

        for key in keyList:
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            neuron_derivatives = dFdt(neurons)
            velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Curvature']

            if key in excludeInterval.keys():
                for interval in excludeInterval[key]:
                    idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                    time = time[idxs]
                    neurons = neurons[:,idxs]
                    neuron_derivatives = neuron_derivatives[:,idxs]
                    velocity = velocity[idxs]
                    curvature = curvature[idxs]

            data[key] = {'time': time, 'neurons': neurons, 'neuron_derivatives': neuron_derivatives, 'velocity': velocity, 'curvature': curvature}

    with open('%s/%s_recordings.dat' % (user_tracker.codePath(), gtype), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
