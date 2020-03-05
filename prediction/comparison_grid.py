import SLM
from Classifier import rectified_derivative
import userTracker
import dataHandler as dh

import numpy as np
from sklearn.decomposition import PCA

import os

results = {}
for typ_cond in ['AKS297.51_moving', 'AML32_moving']:
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

    for key in filter(lambda x: x[-6:] in ('110803', '105254', '105620', '134800'), keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
        curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

        nderiv_pos, nderiv_neg = rectified_derivative(neurons)
        nderivs = np.vstack((nderiv_pos, nderiv_neg))

        neurons_and_derivs = np.vstack((neurons, nderivs))

        pca = PCA(n_components=3)
        pca.fit(neurons.T)
        neurons_reduced = pca.transform(neurons.T).T

        pcderiv_pos, pcderiv_neg = rectified_derivative(neurons_reduced)
        pcderivs = np.vstack((pcderiv_pos, pcderiv_neg))

        print('\tBest Neuron')
        bsn = SLM.optimize_slm(time, neurons, velocity, options = {'best_neuron': True})
        bsn_deriv = SLM.optimize_slm(time, nderivs, velocity, options = {'best_neuron': True})

        print('\tBest PC')
        pc = SLM.optimize_slm(time, neurons_reduced, velocity, options = {'best_neuron': True})
        pc_deriv = SLM.optimize_slm(time, pcderivs, velocity, options = {'best_neuron': True})

        print('\tLinear Model')
        slm = SLM.optimize_slm(time, neurons, velocity)
        slm_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, velocity)
        slm_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {'derivative_penalty': True})

        print('\tLinear Model w/ Classifier')
        slm_tree = SLM.optimize_slm(time, neurons, velocity, options={'decision_tree': True})
        slm_tree_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, velocity, options={'decision_tree': True})
        slm_tree_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, velocity, options={'decision_tree': True, 'derivative_penalty': True})

        results[key] = {'bsn': bsn, 
                        'bsn_deriv': bsn_deriv, 
                        'pc': pc, 
                        'pc_deriv': pc_deriv, 
                        'slm': slm, 
                        'slm_with_derivs': slm_with_derivs, 
                        'slm_with_derivs_acc': slm_with_derivs_acc, 
                        'slm_tree': slm_tree, 
                        'slm_tree_with_derivs': slm_tree_with_derivs, 
                        'slm_tree_with_derivs_acc': slm_tree_with_derivs_acc
                        }

import pickle
with open('comparison_results.dat', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)