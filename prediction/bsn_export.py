import matplotlib.pyplot as plt
import numpy as np
import pickle

from Classifier import rectified_derivative

with open('new_comparison.dat', 'rb') as handle:
    data = pickle.load(handle)#, encoding = 'bytes')

with open('neuron_data_bothmc_nb.dat', 'rb') as f:
    neuron_data = pickle.load(f)#, encoding = 'bytes')

bsns = {}
for k in data.keys():
    vel_idx = np.argmax(np.abs(data[k]['velocity'][True]['weights']))
    curv_idx = np.argmax(np.abs(data[k]['curvature'][True]['weights']))

    _, _, nderiv = rectified_derivative(neuron_data[k]['neurons'])
    neurons_and_derivs = np.vstack((neuron_data[k]['neurons'], nderiv))

    bsns[k] = {'velocity':  {'neuron': neurons_and_derivs[vel_idx,:],  'signal': data[k]['velocity'][True]['signal'],  'time': neuron_data[k]['time'], 'population_fit': data[k]['velocity'][False]['output']},
               'curvature': {'neuron': neurons_and_derivs[curv_idx,:], 'signal': data[k]['curvature'][True]['signal'], 'time': neuron_data[k]['time'], 'population_fit': data[k]['curvature'][False]['output']}}


with open('bsns.dat', 'wb') as f:
    pickle.dump(bsns, f)