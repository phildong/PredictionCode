import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.backends.backend_pdf

import numpy as np
from scipy.ndimage import gaussian_filter
from utility import user_tracker
from utility import data_handler as dh

import os
import pickle

with open('%s/gcamp_linear_models.dat' % user_tracker.codePath(), 'rb') as handle:
    data = pickle.load(handle)

with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as handle:
    neuron_data = pickle.load(handle)

behaviors = ['velocity', 'curvature']

outfilename = 'weights_vel_vs_curv.pdf'

outputFolder = os.path.join(user_tracker.codePath(),'figures/output')
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, outfilename))

for key in data.keys():

    slm_weights_raw = [None] * 2
    slm_weights_raw_deriv = [None] * 2

    for k, behavior in enumerate(behaviors):
        nneurons = data[key][behavior][False]['weights'].size / 2
        slm_weights_raw[k] = np.abs(data[key][behavior][False]['weights'][:nneurons])
        slm_weights_raw_deriv[k] = np.abs(data[key][behavior][False]['weights'][nneurons:])

    fig1 = plt.figure(constrained_layout=True, figsize=[10, 10])
    fig1.suptitle(key)
    #find smallest weight and largest weight
    allweights = np.concatenate([np.concatenate(slm_weights_raw_deriv), np.concatenate(slm_weights_raw)])
    lw = np.max(allweights)


    #Generate scatter plot and box plot
    f1_ax1 = fig1.add_subplot(111, xlabel='Magnitude of Weight for ' + behaviors[1], ylabel='Magnitude of Weight for '+ behaviors[0])
    f1_ax1.plot(slm_weights_raw[1], slm_weights_raw[0], 'o', label='F')
    f1_ax1.plot(slm_weights_raw_deriv[1], slm_weights_raw_deriv[0], 'o', label='dF/dt')
    if key == 'BrainScanner20200130_110803':
        AVAR = 32
        AVAL = 15
        f1_ax1.text(slm_weights_raw[1][AVAR], slm_weights_raw[0][AVAR], 'AVAR')
        f1_ax1.text(slm_weights_raw[1][AVAL], slm_weights_raw[0][AVAR], 'AVAL')
        f1_ax1.text(slm_weights_raw_deriv[1][AVAL], slm_weights_raw_deriv[0][AVAL], 'AVAL')
        f1_ax1.text(slm_weights_raw_deriv[1][AVAR], slm_weights_raw_deriv[0][AVAR], 'AVAR')

    f1_ax1.tick_params(labelsize=18)
    fig1.legend()
    pdf.savefig(fig1)

    fig2 = plt.figure(constrained_layout=True, figsize=[10, 10])
    fig2.suptitle('%s max(F, dF/dt)'% key)
    f2_ax1 = fig2.add_subplot(111, xlabel='Magnitude of Max Weight for ' + behaviors[1], ylabel='Magnitude of Max Weight for '+ behaviors[0])
    maxw =[np.max([[slm_weights_raw[0]], [slm_weights_raw_deriv[0]]], axis=0).T,  np.max([[slm_weights_raw[1]], [slm_weights_raw_deriv[1]]], axis=0).T]
    f2_ax1.plot(maxw[1], maxw[0], 'o', label='max(F,dF/dt)')
    if key == 'BrainScanner20200130_110803':
        AVAR = 32
        AVAL = 15
        f2_ax1.text(maxw[1][AVAR], maxw[0][AVAR], 'AVAR')
        f2_ax1.text(maxw[1][AVAL], maxw[0][AVAL], 'AVAL')
    f2_ax1.tick_params(labelsize=18)
    pdf.savefig(fig2)

pdf.close()
print("wrote " + outfilename)

