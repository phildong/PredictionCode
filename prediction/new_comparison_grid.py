import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

with open('neuron_data_bothmc_nb_aml18.dat', 'rb') as f:
    data = pickle.load(f)
    
results = {}

for key in data.keys():
    print("Running "+key)
    time = data[key]['time']
    neurons = data[key]['neurons']
    neurons_ica = data[key]['neurons_ica']
    velocity = data[key]['velocity']
    curvature = data[key]['curvature']
    phasevelocity = data[key]['phase_velocity']
    grosscurvature = data[key]['gross_curvature']

    _, _, nderiv = rectified_derivative(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))
    _, _, nderiv_ica = rectified_derivative(neurons_ica)
    neurons_and_derivs_ica = np.vstack((neurons_ica, nderiv_ica))

    results[key] = {}
    results[key]['velocity'] = {}
    results[key]['curvature'] = {}
    for ica in [True, False]:
        for nb in [True, False]:
            for bsn in [True, False]:
                print(ica, nb, bsn)
                neur = neurons_and_derivs_ica if ica else neurons_and_derivs
                results[key]['velocity'][(ica, nb, bsn)] = SLM.optimize_slm(time, neur, phasevelocity if nb else velocity, options = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
                results[key]['curvature'][(ica, nb, bsn)] = SLM.optimize_slm(time, neur, grosscurvature if nb else curvature, options = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
    
with open('new_comparison_aml18.dat', 'wb') as f:
    pickle.dump(results, f)