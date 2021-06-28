import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "linear.pdf"))
pdf_aml18 = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "linear_aml18.pdf"))

with open('neuron_data_linear_rfp.dat', 'rb') as f:
    data = pickle.load(f)
    
with open('neuron_data_linear_rfp_aml18.dat', 'rb') as f:
    data_aml18 = pickle.load(f)

for key in data.keys():
    print("Running "+key)
    time = data[key]['time']
    neurons = data[key]['neurons']
    velocity = data[key]['velocity']
    curvature = data[key]['curvature']

    _, _, nderiv = rectified_derivative(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))

    print("Running velocity Linear")
    res = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Linear velocity  R2_ms = %0.3f' % res['scorespredicted'][1])
    print('')

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(key)
    ax[0].set_ylabel('Velocity', fontsize=16)
    ax[0].plot(res['time'], res['signal'], 'k', lw=1)
    ax[0].plot(res['time'], res['output'], 'b', lw=1)
    ax[0].set_xlabel('Time (s)', fontsize=16)
    ax[0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
    ax[0].set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)

    # pdf.savefig(fig)

    print("Running curvature Linear")
    res = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Linear velocity  R2_ms = %0.3f' % res['scorespredicted'][1])
    print('')

    ax[1].set_ylabel('Curvature', fontsize=16)
    ax[1].plot(res['time'], res['signal'], 'k', lw=1)
    ax[1].plot(res['time'], res['output'], 'b', lw=1)
    ax[1].set_xlabel('Time (s)', fontsize=16)
    ax[1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
    ax[1].set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    
    pdf.savefig(fig)
    plt.close('all')

pdf.close()

for key in data_aml18.keys():
    print("Running "+key)
    time = data_aml18[key]['time']
    neurons = data_aml18[key]['neurons']
    velocity = data_aml18[key]['velocity']
    curvature = data_aml18[key]['curvature']

    _, _, nderiv = rectified_derivative(neurons)
    neurons_and_derivs = np.vstack((neurons, nderiv))

    print("Running velocity Linear")
    res = SLM.optimize_slm(time, neurons_and_derivs, velocity, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Linear velocity  R2_ms = %0.3f' % res['scorespredicted'][1])
    print('')

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(key)
    ax[0].set_ylabel('Velocity', fontsize=16)
    ax[0].plot(res['time'], res['signal'], 'k', lw=1)
    ax[0].plot(res['time'], res['output'], 'b', lw=1)
    ax[0].set_xlabel('Time (s)', fontsize=16)
    ax[0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
    ax[0].set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)

    # pdf.savefig(fig)

    print("Running curvature Linear")
    res = SLM.optimize_slm(time, neurons_and_derivs, curvature, options = {"l1_ratios": [0], "parallelize": False})
    
    print('Linear velocity  R2_ms = %0.3f' % res['scorespredicted'][1])
    print('')

    ax[1].set_ylabel('Curvature', fontsize=16)
    ax[1].plot(res['time'], res['signal'], 'k', lw=1)
    ax[1].plot(res['time'], res['output'], 'b', lw=1)
    ax[1].set_xlabel('Time (s)', fontsize=16)
    ax[1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)
    ax[1].set_title(r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    
    pdf_aml18.savefig(fig)
    plt.close('all')

pdf_aml18.close()

# pdf_summary = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "linear_vs_rfp_summary.pdf"))

# for v, c in [[vel_pairs, vel_pairs_aml18], [curv_pairs, curv_pairs_aml18]]:
#     fig, ax = plt.subplots(1, 2, figsize = (20, 7))
#     ax[0].set_title('GCaMP')
#     ax[0].set_xticks([0, 1])
#     ax[0].set_xticklabels(['ICA', 'RFP'])
#     ax[0].set_ylim([-0.5, 1])
#     ax[0].set_yticks([-0.5, -.25, 0, .25, .5, .75, 1])
#     ax[0].grid(axis='x', linestyle='--')

#     ax[1].set_title('GFP')
#     ax[1].set_xticks([0, 1])
#     ax[1].set_xticklabels(['ICA', 'RFP'])
#     ax[1].set_ylim([-0.5, 1])
#     ax[1].set_yticks([-0.5, -.25, 0, .25, .5, .75, 1])
#     ax[1].grid(axis='x', linestyle='--')

#     for p in v:
#         ax[0].plot(p)
#     for p in c:
#         ax[1].plot(p)

#     pdf_summary.savefig(fig)

# pdf_summary.close()

        