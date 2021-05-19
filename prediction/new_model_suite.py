import SLM
import MARS
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

for key in filter(lambda x: x[-6:] == '110803', data.keys()):
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

    print("\t main")
    results[key]['velocity']['main'] = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False})
    results[key]['curvature']['main'] = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False})
    print(results[key]['velocity']['main']['scorespredicted'][1], results[key]['curvature']['main']['scorespredicted'][1])

    print("\t no_deriv")
    results[key]['velocity']['no_deriv'] = SLM.optimize_slm(time, neurons, velocity, options = {"l1_ratios": [0], "parallelize": False})
    results[key]['curvature']['no_deriv'] = SLM.optimize_slm(time, neurons, curvature, options = {"l1_ratios": [0], "parallelize": False})
    print(results[key]['velocity']['no_deriv']['scorespredicted'][1], results[key]['curvature']['no_deriv']['scorespredicted'][1])

    print("\t acc")
    results[key]['velocity']['acc'] = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    results[key]['curvature']['acc'] = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    print(results[key]['velocity']['acc']['scorespredicted'][1], results[key]['curvature']['acc']['scorespredicted'][1])

    print("\t no_deriv_acc")
    results[key]['velocity']['no_deriv_acc'] = SLM.optimize_slm(time, neurons, velocity, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    results[key]['curvature']['no_deriv_acc'] = SLM.optimize_slm(time, neurons, curvature, options = {"l1_ratios": [0], "parallelize": False, "derivative_penalty": True})
    print(results[key]['velocity']['no_deriv_acc']['scorespredicted'][1], results[key]['curvature']['no_deriv_acc']['scorespredicted'][1])

    print("\t l0.01")
    results[key]['velocity']['l0.01'] = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0.01], "parallelize": False})
    results[key]['curvature']['l0.01'] = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0.01], "parallelize": False})
    print(results[key]['velocity']['l0.01']['scorespredicted'][1], results[key]['curvature']['l0.01']['scorespredicted'][1])

    print("\t no_deriv_l0.01")
    results[key]['velocity']['no_deriv_l0.01'] = SLM.optimize_slm(time, neurons, velocity, options = {"l1_ratios": [0.01], "parallelize": False})
    results[key]['curvature']['no_deriv_l0.01'] = SLM.optimize_slm(time, neurons, curvature, options = {"l1_ratios": [0.01], "parallelize": False})
    print(results[key]['velocity']['no_deriv_l0.01']['scorespredicted'][1], results[key]['curvature']['no_deriv_l0.01']['scorespredicted'][1])

    print("\t tree")
    results[key]['velocity']['tree'] = SLM.optimize_slm(time, neurons, velocity, options = {"l1_ratios": [0], "decision_tree": True, "parallelize": False})
    results[key]['curvature']['tree'] = SLM.optimize_slm(time, neurons, curvature, options = {"l1_ratios": [0], "decision_tree": True, "parallelize": False})
    # print(results[key]['velocity']['tree']['scorespredicted'][1], results[key]['curvature']['tree']['scorespredicted'][1])

    print("\t mars")
    results[key]['velocity']['mars'] = MARS.optimize_mars(time, neurons, velocity)
    results[key]['curvature']['mars'] = MARS.optimize_mars(time, neurons, curvature)
    print(results[key]['velocity']['mars']['scorespredicted'][1], results[key]['curvature']['mars']['scorespredicted'][1])
    
with open('new_model_suite.dat', 'wb') as f:
    pickle.dump(results, f)