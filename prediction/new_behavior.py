import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "new_behavior_old_4-21.pdf"))

vel_pairs = []
curv_pairs = []

with open('neuron_data_old.dat', 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    print("Running "+key)
    time = data[key]['time']
    neurons = data[key]['neurons']
    velocity = data[key]['velocity']
    curvature = data[key]['curvature']
    # phasevelocity = data[key]['phase_velocity']
    # grosscurvature = data[key]['gross_curvature']

    _, _, nderiv = rectified_derivative(neurons)

    neurons_and_derivs = np.vstack((neurons, nderiv))

    print("Running old velocity")
    old = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False})
    # print("Running new velocity")
    # new = SLM.optimize_slm(time, neurons_and_derivs, phasevelocity, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Eigenworm velocity  R2_ms = %0.3f' % old['scorespredicted'][1])
    # print('Phase velocity      R2_ms = %0.3f' % new['scorespredicted'][1])
    print('')
    # vel_pairs.append([old['scorespredicted'][1], new['scorespredicted'][1]])

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(key)
    axs[0].set_title('Eigenworm Velocity', fontsize=20)
    axs[1].set_title('Phase Shift Velocity', fontsize=20)

    for res, ax in zip([old, old], axs):
        ax.plot(res['time'], res['signal'], 'k', lw=1)
        ax.plot(res['time'], res['output'], 'b', lw=1)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Velocity', fontsize=16)
        ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
        ax.set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    
    pdf.savefig(fig)

    print("Running old curvature")
    oldcurv = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False})
    # print("Running new curvature")
    # newcurv = SLM.optimize_slm(time, neurons_and_derivs, grosscurvature, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Eigenworm curvature R2_ms = %0.3f' % oldcurv['scorespredicted'][1])
    # print('Gross curvature     R2_ms = %0.3f' % newcurv['scorespredicted'][1])
    print('')
    # curv_pairs.append([oldcurv['scorespredicted'][1], newcurv['scorespredicted'][1]])

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(key)
    axs[0].set_title('Eigenworm Curvature', fontsize=20)
    axs[1].set_title('Geometric Curvature', fontsize=20)

    for res, ax in zip([oldcurv, oldcurv], axs):
        ax.plot(res['time'], res['signal'], 'k', lw=1)
        ax.plot(res['time'], res['output'], 'b', lw=1)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Curvature', fontsize=16)
        ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
        ax.set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    
    pdf.savefig(fig)
    plt.close('all')

pdf.close()

# fig, ax = plt.subplots(1, 2, figsize = (20, 7))
# for p in vel_pairs:
#     ax[0].plot(p)
# for p in curv_pairs:
#     ax[1].plot(p)

# fig.savefig('new_behavior_overall.pdf')

        