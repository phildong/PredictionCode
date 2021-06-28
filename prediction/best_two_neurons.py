import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "best_two_neurons.pdf"))

with open('neuron_data.dat', 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    print("Running "+key)
    time = data[key]['time']
    neurons = data[key]['neurons']
    velocity = data[key]['velocity']
    curvature = data[key]['curvature']

    _, _, nderiv = rectified_derivative(neurons)

    neurons_and_derivs = np.vstack((neurons, nderiv))

    bsn_deriv = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {'best_neuron': True})
    
    print('BSN  R2_ms = %0.3f' % bsn_deriv['scorespredicted'][1])

    bsn_deriv_pos = SLM.optimize_slm(time[velocity > 0], neurons_and_derivs[:,velocity > 0], velocity[velocity > 0], options = {'best_neuron': True})
    bsn_deriv_neg = SLM.optimize_slm(time[velocity < 0], neurons_and_derivs[:,velocity < 0], velocity[velocity < 0], options = {'best_neuron': True})

    i1 = np.nonzero(bsn_deriv_pos['weights'])[0][0]
    i2 = np.nonzero(bsn_deriv_neg['weights'])[0][0]

    print("I used: %d, %d" % (i1 % neurons.shape[0], i2 % neurons.shape[0]))

    best_two = SLM.optimize_slm(time, neurons_and_derivs[[i1,i2],:], velocity)

    print('BSN2 R2_ms = %0.3f' %  best_two['scorespredicted'][1])
    print("")

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(key)

    for res, ax in zip([bsn_deriv, best_two], axs):
        ax.plot(res['time'], res['signal'], 'k', lw=1)
        ax.plot(res['time'], res['output'], 'b', lw=1)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Velocity', fontsize=16)
        ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
        ax.set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    
    pdf.savefig(fig)

pdf.close()

        