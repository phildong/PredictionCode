import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle

def rho_adj(y, yhat):
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    rho2 = np.corrcoef(y, yhat)[0,1]**2

    return rho2 - (alpha)**2/((truesigma*predsigma)**2)

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

pickled_data = 'comparison_results_velocity_l10.dat'
with open(pickled_data, 'rb') as handle:
    vel_data = pickle.load(handle, encoding = 'bytes')

pickled_data = 'comparison_results_curvature_l10.dat'
with open(pickled_data, 'rb') as handle:
    curv_data = pickle.load(handle, encoding = 'bytes')

with open('neuron_data.dat', 'rb') as f:
    neuron_data = pickle.load(f, encoding = 'bytes')

fig, ax = plt.subplots(2, 4, figsize = (25, 15))
keys = sorted(vel_data.keys(), key = lambda x: -vel_data[x][b'slm_with_derivs'][b'scorespredicted'][2])

for ii, dataset in enumerate(keys[:8]):
    print(dataset)
    row = ii // 4
    col = ii % 4

    vel_res = vel_data[dataset][b'slm_with_derivs']   
    curv_res = curv_data[dataset][b'slm_with_derivs']   

    neurons_unn = neuron_data[dataset][b'neurons']
    _, _, nderiv = rectified_derivative(neurons_unn)
    neurons_and_derivs = np.vstack((neurons_unn, nderiv))

    mean = np.mean(neurons_and_derivs, axis = 1)[:, np.newaxis]
    std = np.std(neurons_and_derivs, axis = 1)[:, np.newaxis]
    neurons = (neurons_and_derivs-mean)/std
    nn = neurons.shape[0]//2

    vel_order = np.argsort(-np.abs(vel_res[b'weights'][:nn])-np.abs(vel_res[b'weights'][nn:]))
    curv_order = np.argsort(-np.abs(curv_res[b'weights'][:nn])-np.abs(curv_res[b'weights'][nn:]))

    vel_lines = np.zeros((vel_order.size, vel_res[b'time'].size))
    curv_lines = np.zeros((curv_order.size, curv_res[b'time'].size))

    for i in range(vel_order.size):
        idxs = vel_order[:i]
        output = vel_res[b'intercepts'] + np.dot(vel_res[b'weights'][idxs], neurons[idxs,:]) + np.dot(vel_res[b'weights'][idxs+nn], neurons[idxs+nn,:])

        coef = np.polyfit(output[vel_res[b'train_idx']],vel_res[b'signal'][vel_res[b'train_idx']],1)
        poly1d_fn = np.poly1d(coef) 

        vel_lines[i,:] = poly1d_fn(output)

    for i in range(curv_order.size):
        idxs = curv_order[:i]
        output = curv_res[b'intercepts'] + np.dot(curv_res[b'weights'][idxs], neurons[idxs,:]) + np.dot(curv_res[b'weights'][idxs+nn], neurons[idxs+nn,:])

        coef = np.polyfit(output[curv_res[b'train_idx']],curv_res[b'signal'][curv_res[b'train_idx']],1)
        poly1d_fn = np.poly1d(coef) 

        curv_lines[i,:] = poly1d_fn(output)

    vel_rhos = np.array([rho_adj(vel_res[b'signal'][vel_res[b'test_idx']], vel_lines[i,:][vel_res[b'test_idx']]) for i in range(vel_order.size)])
    curv_rhos = np.array([rho_adj(curv_res[b'signal'][curv_res[b'test_idx']], curv_lines[i,:][curv_res[b'test_idx']]) for i in range(curv_order.size)])

    ax[row][col].set_title(dataset[12:])
    ax[row][col].set_xlim((0, 1))
    ax[row][col].set_ylabel('Number of Neurons')
    ax[row][col].set_xlabel(r'Fraction of full model performance')

    thresh_vals = np.linspace(0, 1, 101)
    vel_counts = np.array([np.argmax(vel_rhos > t*vel_rhos[-1]) for t in thresh_vals])
    curv_counts = np.array([np.argmax(curv_rhos > t*curv_rhos[-1]) for t in thresh_vals])

    overlap_counts = np.array([np.intersect1d(vel_order[:vel_counts[i]],curv_order[:curv_counts[i]]).size for i in range(vel_counts.size)])

    ax[row][col].plot(thresh_vals, vel_counts, color='blue')
    ax[row][col].fill_between(thresh_vals, np.zeros(vel_counts.size), vel_counts, color='blue', alpha=0.2, label = 'Velocity')

    ax[row][col].plot(thresh_vals, curv_counts, color='green')
    ax[row][col].fill_between(thresh_vals, np.zeros(curv_counts.size), curv_counts, color='green', alpha=0.2, label = 'Curvature')

    ax[row][col].plot(thresh_vals, overlap_counts, color='red')
    ax[row][col].fill_between(thresh_vals, np.zeros(overlap_counts.size), overlap_counts, color='red', alpha=0.2, label = 'Both')

    ax[row][col].legend()

fig.tight_layout(pad=3, w_pad=4, h_pad=4)
fig.savefig('highly_weighted.png')