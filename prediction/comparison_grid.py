import SLM
from Classifier import rectified_derivative
import userTracker
import dataHandler as dh

import numpy as np
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter1d

import os

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

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

    for key in keyList:
        if key in excludeSets:
            continue
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
                velocity = velocity[idxs]
                # cmsvelocity = cmsvelocity[idxs]
                curvature = curvature[idxs]

        _, _, nderiv = rectified_derivative(neurons)

        neurons_and_derivs = np.vstack((neurons, nderiv))

        pca = PCA(n_components=3)
        pca.fit(neurons.T)
        neurons_reduced = pca.transform(neurons.T).T

        _, _, pcderiv = rectified_derivative(neurons_reduced)

        pc_and_derivs = np.vstack((neurons_reduced, pcderiv))

        print('\tCenterline Velocity')
        print('\t\tBest Neuron')
        bsn = SLM.optimize_slm(time, neurons, velocity, options = {'best_neuron': True})
        bsn_deriv = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {'best_neuron': True})
        bsn_deriv_acc = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {'best_neuron': True, 'derivative_penalty': True})

        print('\t\tBest PC')
        pc = SLM.optimize_slm(time, neurons_reduced, velocity, options = {'best_neuron': True})
        pc_deriv = SLM.optimize_slm(time, pc_and_derivs, velocity, options = {'best_neuron': True})

        print('\t\tLinear Model')
        # slm = SLM.optimize_slm(time, neurons, velocity)
        # slm_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, velocity)
        slm_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {'derivative_penalty': True})

        # print('\tLinear Model w/ Classifier')
        # slm_tree = SLM.optimize_slm(time, neurons, velocity, options={'decision_tree': True})
        # slm_tree_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, velocity, options={'decision_tree': True})
        # slm_tree_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, velocity, options={'decision_tree': True, 'derivative_penalty': True})

        # print('\tMARS')
        # mars = MARS.optimize_mars(time, neurons, velocity, options={'max_terms': 5 if key[:3] == 'AML' else 2})

        results[key] = {'bsn': bsn, 
                        'bsn_deriv': bsn_deriv, 
                        'bsn_deriv_acc': bsn_deriv_acc,
                        'pc': pc, 
                        'pc_deriv': pc_deriv, 
                        # 'slm': slm, 
                        # 'slm_with_derivs': slm_with_derivs, 
                        'slm_with_derivs_acc': slm_with_derivs_acc, 
                        # 'slm_tree': slm_tree, 
                        # 'slm_tree_with_derivs': slm_tree_with_derivs, 
                        # 'slm_tree_with_derivs_acc': slm_tree_with_derivs_acc,
                        # 'mars': mars
                        }

        # print('\tCMS Velocity')
        # print('\t\tBest Neuron')
        # bsn = SLM.optimize_slm(time, neurons, cmsvelocity, options = {'best_neuron': True})
        # bsn_deriv = SLM.optimize_slm(time, neurons_and_derivs, cmsvelocity, options = {'best_neuron': True})
        # bsn_deriv_acc = SLM.optimize_slm(time, neurons_and_derivs, cmsvelocity, options = {'best_neuron': True, 'derivative_penalty': True})

        # print('\t\tBest PC')
        # pc = SLM.optimize_slm(time, neurons_reduced, cmsvelocity, options = {'best_neuron': True})
        # pc_deriv = SLM.optimize_slm(time, pc_and_derivs, cmsvelocity, options = {'best_neuron': True})

        # print('\t\tLinear Model')
        # # slm = SLM.optimize_slm(time, neurons, velocity)
        # # slm_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, velocity)
        # slm_with_derivs_acc = SLM.optimize_slm(time, neurons_and_derivs, cmsvelocity, options = {'derivative_penalty': True})

        # cmsresults[key] = { 'bsn': bsn, 
        #                     'bsn_deriv': bsn_deriv, 
        #                     'bsn_deriv_acc': bsn_deriv_acc,
        #                     'pc': pc, 
        #                     'pc_deriv': pc_deriv, 
        #                     # 'slm': slm, 
        #                     # 'slm_with_derivs': slm_with_derivs, 
        #                     'slm_with_derivs_acc': slm_with_derivs_acc, 
        #                     # 'slm_tree': slm_tree, 
        #                     # 'slm_tree_with_derivs': slm_tree_with_derivs, 
        #                     # 'slm_tree_with_derivs_acc': slm_tree_with_derivs_acc,
        #                     # 'mars': mars
        #                     }

import pickle
with open('comparison_results_aml18.dat', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('comparison_results_cms_smooth.dat', 'wb') as handle:
#     pickle.dump(cmsresults, handle, protocol=pickle.HIGHEST_PROTOCOL)