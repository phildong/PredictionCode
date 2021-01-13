import SLM
import MARS
from Classifier import rectified_derivative
import userTracker
import dataHandler as dh

import numpy as np
from sklearn.decomposition import PCA

import os

data = {}
res = {}
for typ_cond in ['AML310_moving', 'AML32_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))
    outLoc = os.path.join(path, 'Analysis/{}_results.hdf5'.format(typ_cond))
    outLocData = os.path.join(path, 'Analysis/{}.hdf5'.format(typ_cond))

    print("Loading " + folder + "...")
    try:
        # load multiple datasets
        dataSets = dh.loadDictFromHDF(outLocData)
        keyList = np.sort(dataSets.keys())
        results = dh.loadDictFromHDF(outLoc)
        # store in dictionary by typ and condition
        
        data[typ_cond] = {}
        data[typ_cond]['dsets'] = keyList
        data[typ_cond]['input'] = dataSets
        data[typ_cond]['analysis'] = results
    except IOError:
        print typ_cond, 'not found.'
        pass
print('Done reading data.')

key = 'AML310_moving'
idn = 'BrainScanner20200130_110803'
dset = data[key]['input'][idn]
centerlines = np.array(dset['CLFull'])
time = np.array(dset['Neurons']['TimeFull'])

def calc_head_angle(centerlines):
    nose_start = 0
    nose_to_neck = 10
    neck_end = 20

    nose_vec = np.diff(centerlines[:, [nose_start, nose_to_neck], :], axis = 1)
    neck_vec = np.diff(centerlines[:, [nose_to_neck, neck_end], :], axis = 1)

    nose_orientation = np.zeros(centerlines.shape[0])
    neck_orientation = np.zeros(centerlines.shape[0])

    for timept in np.arange(centerlines.shape[0]):
        nose_orientation[timept] = np.arctan2(nose_vec[timept,:,0], nose_vec[timept,:,1])
        neck_orientation[timept] = np.arctan2(neck_vec[timept,:,0], neck_vec[timept,:,1])

    head_angle = nose_orientation - neck_orientation
    return head_angle


head_angle = np.unwrap(calc_head_angle(centerlines))
sigma = 3
smooth_head_angle = dh.gauss_filterNaN(head_angle, sigma)



from scipy.signal import find_peaks
peaks, _ = find_peaks(smooth_head_angle, height = 0, prominence = 0.4)
neg_peaks, _ = find_peaks(smooth_head_angle*-1, height = 0, prominence = 0.4)

def find_phase(peaks, time):
    #Use interpolation

    cum_phase_pks = 2*np.pi * np.arange(peaks.shape[0])

    from scipy import interpolate
    f = interpolate.interp1d(time[peaks], cum_phase_pks, fill_value="extrapolate")

    cum_phase = f(time)

    phase = np.mod(cum_phase, 2*np.pi)

    return phase

phase = np.cos(find_phase(peaks, time)[dset['Neurons']['I_valid_map']])
neg_phase = find_phase(neg_peaks, time)[dset['Neurons']['I_valid_map']]

time = time[dset['Neurons']['I_valid_map']]

neurons = dset['Neurons']['I_smooth_interp_crop_noncontig']

nderiv_pos, nderiv_neg = rectified_derivative(neurons)
nderivs = np.vstack((nderiv_pos, nderiv_neg))

neurons_and_derivs = np.vstack((neurons, nderivs))

pca = PCA(n_components = 3)
pca.fit(neurons.T)
neurons_reduced = pca.transform(neurons.T).T

pcderiv_pos, pcderiv_neg = rectified_derivative(neurons_reduced)
pcderivs = np.vstack((pcderiv_pos, pcderiv_neg))

pc_and_derivs = np.vstack((neurons_reduced, pcderivs))

print('\tBest Neuron')
bsn = SLM.optimize_slm(time, neurons, phase, options = {'best_neuron': True})
bsn_deriv = SLM.optimize_slm(time, neurons_and_derivs, phase, options = {'best_neuron': True})

print('\tBest PC')
pc = SLM.optimize_slm(time, neurons_reduced, phase, options = {'best_neuron': True})
pc_deriv = SLM.optimize_slm(time, pc_and_derivs, phase, options = {'best_neuron': True})

print('\tLinear Model')
slm = SLM.optimize_slm(time, neurons, phase)
slm_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, phase)
slm_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, phase, options = {'derivative_penalty': True})

print('\tLinear Model w/ Classifier')
slm_tree = SLM.optimize_slm(time, neurons, phase, options={'decision_tree': True})
slm_tree_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, phase, options={'decision_tree': True})
slm_tree_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, phase, options={'decision_tree': True, 'derivative_penalty': True})

print('\tMARS')
mars = MARS.optimize_mars(time, neurons, phase)

res[idn] = {'bsn': bsn, 
                'bsn_deriv': bsn_deriv, 
                'pc': pc, 
                'pc_deriv': pc_deriv, 
                'slm': slm, 
                'slm_with_derivs': slm_with_derivs, 
                'slm_with_derivs_acc': slm_with_derivs_acc, 
                'slm_tree': slm_tree, 
                'slm_tree_with_derivs': slm_tree_with_derivs, 
                'slm_tree_with_derivs_acc': slm_tree_with_derivs_acc,
                'mars': mars
                }

import pickle
with open('comparison_results_head_bend.dat', 'wb') as handle:
    pickle.dump(res, handle, protocol = pickle.HIGHEST_PROTOCOL)