import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
from scipy.signal import find_peaks, peak_widths

def rectified_derivative(neurons):
    nan_zero = np.copy(neurons)
    nan_zero[np.isnan(neurons)] = 0
    nan_zero_filtered = gaussian_filter(nan_zero, order = 1, sigma = (0, 7))

    flat = 0*neurons.copy()+1
    flat[np.isnan(neurons)] = 0
    flat_filtered = gaussian_filter(flat, order = 0, sigma = (0, 7))

    deriv = nan_zero_filtered/flat_filtered
    deriv_pos = np.copy(deriv)
    deriv_neg = np.copy(deriv)
    deriv_pos[deriv < 0] = 0
    deriv_neg[deriv > 0] = 0

    return deriv_pos, deriv_neg, deriv

with open('new_comparison.dat', 'rb') as handle:
    data = pickle.load(handle)#, encoding = 'bytes')

with open('neuron_data_bothmc_nb.dat', 'rb') as f:
    neuron_data = pickle.load(f)#, encoding = 'bytes')

for k in filter(lambda x: '134800' in x, data.keys()):
    fig, ax = plt.subplots(6, 2, figsize = (20, 15), sharex = True)

    neurons_unn = neuron_data[k]['neurons']
    _, _, nderiv = rectified_derivative(neurons_unn)
    neurons_and_derivs = np.vstack((neurons_unn, nderiv))

    mean = np.mean(neurons_and_derivs, axis = 1)[:, np.newaxis]
    std = np.std(neurons_and_derivs, axis = 1)[:, np.newaxis]
    neurons = (neurons_and_derivs-mean)/std

    time = data[k]['velocity'][False]['time']
    vel = data[k]['velocity'][False]['signal']
    curv = data[k]['curvature'][False]['signal']

    ax[0,0].set_ylabel('Velocity (mm/s)', fontsize = 14)
    ax[0,1].set_ylabel('Curvature', fontsize = 14)

    for col, behstr in enumerate(['velocity', 'curvature']):

        beh = data[k][behstr][False]['signal']

        ax[0,col].plot(time, beh, color='k', linewidth = 2)
        ax[5,col].set_xlabel('Time (s)', fontsize = 14)

        weights = data[k][behstr][False]['weights']
        hws = np.argsort(-np.abs(weights))
        nn = weights.size//2

        for i in range(5):
            ax[i+1,col].plot(time, neurons[hws[i],:], color = '#ff8c00' if hws[i] >= nn else '#008cff')
            ax[i+1,col].set_ylabel('#%d (W = %0.3f)' % (hws[i] % nn, weights[hws[i]]), color = '#ff8c00' if hws[i] >= nn else '#008cff', fontsize = 14)

        peaks_pos, _ = find_peaks(beh, prominence=(np.max(beh)-np.min(beh))/2)
        peaks_neg, _ = find_peaks(-beh, prominence=(np.max(beh)-np.min(beh))/2)
        peaks = np.hstack((peaks_pos, peaks_neg))
        
        _, _, lpos, rpos = peak_widths(beh, peaks_pos, rel_height = 0.25-.1*(1-col))
        _, _, lneg, rneg = peak_widths(-beh, peaks_neg, rel_height = 0.25-.1*col)
        lips = np.hstack((lpos, lneg))
        rips = np.hstack((rpos, rneg))

        for a, b in zip(lips, rips):
            ax[0,col].fill_between([time[int(a)], time[int(b+1)]], [min(beh), min(beh)], [max(beh), max(beh)], alpha = 0.2, color = 'red')
            for i in range(5):
                ax[i+1,col].fill_between([time[int(a)], time[int(b+1)]], [min(neurons[hws[i],:]), min(neurons[hws[i],:])], [max(neurons[hws[i],:]), max(neurons[hws[i],:])], alpha = 0.2, color = 'red')

        ax[0,col].plot(time[peaks_pos], beh[peaks_pos], "vb", markersize = 10)
        ax[0,col].plot(time[peaks_neg], beh[peaks_neg], "^b", markersize = 10)

    fig.suptitle(k, fontsize = 18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('fig5_%s.png' % k)