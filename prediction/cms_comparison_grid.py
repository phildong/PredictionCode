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
    velocity = data[key]['cmsvelocity']

    velz = (velocity - np.mean(velocity))/np.std(velocity)
    inc = np.where(np.abs(velz) <= 2)[0]
    print(velz.size-inc.size)

    time = time[inc]
    neurons = neurons[:,inc]
    neurons_ica = neurons_ica[:,inc]
    velocity = gaussian_filter1d(velocity[inc], sigma = 7)

    _, _, nderiv = rectified_derivative(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))
    _, _, nderiv_ica = rectified_derivative(neurons_ica)
    neurons_and_derivs_ica = np.vstack((neurons_ica, nderiv_ica))

    results[key] = {}
    for ica in [True, False]:
        for bsn in [True, False]:
            print(ica, bsn)
            neur = neurons_and_derivs_ica if ica else neurons_and_derivs
            results[key][(ica, bsn)] = SLM.optimize_slm(time, neur, velocity, options = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
            print(results[key][(ica, bsn)]['scorespredicted'][1])
    
with open('new_comparison_cms_aml18.dat', 'wb') as f:
    pickle.dump(results, f)