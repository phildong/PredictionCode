import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

with open('neuron_data_bothmc_nb.dat', 'rb') as f:
    data = pickle.load(f)
    
results = {}

for key in data.keys():
    print("Running "+key)
    time = data[key]['time']
    neurons = data[key]['neurons']
    velocity = data[key]['cmsvelocity']
    curvature = data[key]['gross_curvature']

    _, _, nderiv = rectified_derivative(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))

    results[key] = {}
    results[key]['velocity'] = {}
    results[key]['curvature'] = {}
    for bsn in [True, False]:
        results[key]['velocity'][bsn] = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
        results[key]['curvature'][bsn] = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
        print(results[key]['velocity'][bsn]['scorespredicted'][1], results[key]['curvature'][bsn]['scorespredicted'][1])
    
with open('new_comparison.dat', 'wb') as f:
    pickle.dump(results, f)