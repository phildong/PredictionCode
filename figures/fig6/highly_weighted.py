import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter

from utility import user_tracker
from prediction.models import linear

import pickle
import os

outputFolder = os.path.join(user_tracker.codePath(),'figures/output')
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

with open('%s/gcamp_linear_models.dat' % user_tracker.codePath(), 'rb') as handle:
    data = pickle.load(handle)

with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as f:
    neuron_data = pickle.load(f)

fig, ax = plt.subplots(3, 4, figsize = (25, 15))
keys = sorted(data.keys(), key = lambda x: -data[x]['velocity'][False]['R2ms_test'])
rho2_adj_vel = np.array([data[keys[x]]['velocity'][False]['R2ms_test'] for x in np.arange(len(keys))])
rho2_adj_curv = np.array([data[keys[x]]['curvature'][False]['R2ms_test'] for x in np.arange(len(keys))])

#preallocate arrays for plotting later
n_impact_vel, n_impact_curv,  n_impact_overlap= np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))

nnn=np.zeros(len(keys))
for ii, dataset in enumerate(keys):
    print(dataset)
    nnn[ii] = neuron_data[dataset]['neurons'].shape[0]
    row = ii // 4
    col = ii % 4

    vel_res = data[dataset]['velocity'][False]
    curv_res = data[dataset]['curvature'][False]

    neurons_and_derivs = np.vstack((neuron_data[dataset]['neurons'], neuron_data[dataset]['neuron_derivatives']))

    mean = np.mean(neurons_and_derivs, axis = 1)[:, np.newaxis]
    std = np.std(neurons_and_derivs, axis = 1)[:, np.newaxis]
    neurons = (neurons_and_derivs-mean)/std
    nn = neurons.shape[0]//2

    vel_order = np.argsort(-np.maximum(np.abs(vel_res[b'weights'][:nn]),np.abs(vel_res[b'weights'][nn:])))
    curv_order = np.argsort(-np.maximum(np.abs(curv_res[b'weights'][:nn]),np.abs(curv_res[b'weights'][nn:])))

    vel_lines = np.zeros((vel_order.size, vel_res[b'time'].size))
    curv_lines = np.zeros((curv_order.size, curv_res[b'time'].size))

    for i in range(vel_order.size):
        idxs = vel_order[:i]
        output = vel_res[b'intercept'] + np.dot(vel_res[b'weights'][idxs], neurons[idxs,:]) + np.dot(vel_res[b'weights'][idxs+nn], neurons[idxs+nn,:])

        coef = np.polyfit(output[vel_res[b'train_idx']],vel_res[b'signal'][vel_res[b'train_idx']],1)
        poly1d_fn = np.poly1d(coef) 

        vel_lines[i,:] = poly1d_fn(output)

    for i in range(curv_order.size):
        idxs = curv_order[:i]
        output = curv_res[b'intercept'] + np.dot(curv_res[b'weights'][idxs], neurons[idxs,:]) + np.dot(curv_res[b'weights'][idxs+nn], neurons[idxs+nn,:])

        coef = np.polyfit(output[curv_res[b'train_idx']],curv_res[b'signal'][curv_res[b'train_idx']],1)
        poly1d_fn = np.poly1d(coef) 

        curv_lines[i,:] = poly1d_fn(output)

    vel_rhos = np.array([linear.R2ms(vel_res[b'signal'], vel_lines[i,:]) for i in range(vel_order.size)])
    curv_rhos = np.array([linear.R2ms(curv_res[b'signal'], curv_lines[i,:]) for i in range(curv_order.size)])

    ax[row][col].set_title(dataset[12:] + ' N=%d' %nn)
    ax[row][col].set_xlim((0, 1))
    ax[row][col].set_ylabel('Number of Neurons')
    ax[row][col].set_xlabel('Fraction of full model performance')
    ax[row][col].tick_params(axis="x", labelsize=16)
    ax[row][col].tick_params(axis="y", labelsize=16)

    thresh_vals = np.linspace(0, 1, 101)
    vel_counts = np.array([np.argmax(vel_rhos > t*vel_rhos[-1]) for t in thresh_vals])
    curv_counts = np.array([np.argmax(curv_rhos > t*curv_rhos[-1]) for t in thresh_vals])
    overlap_counts = np.array([np.intersect1d(vel_order[:vel_counts[i]],curv_order[:curv_counts[i]]).size for i in range(vel_counts.size)])

    impact_thresh = 0.9
    n_impact_vel[ii] = np.interp(impact_thresh, thresh_vals, vel_counts)
    n_impact_curv[ii] = np.interp(impact_thresh, thresh_vals, curv_counts)
    n_impact_overlap[ii] = np.interp(impact_thresh, thresh_vals, overlap_counts)

    ax[row][col].set_title(dataset[12:] + ' N=%d, (%d, %d, %d)' %(nn, n_impact_vel[ii], n_impact_curv[ii], n_impact_overlap[ii]))
    ax[row][col].plot(thresh_vals, vel_counts, color='blue')
    ax[row][col].fill_between(thresh_vals, np.zeros(vel_counts.size), vel_counts, color='blue', alpha=0.2, label = 'Velocity')

    ax[row][col].plot(thresh_vals, curv_counts, color='green')
    ax[row][col].fill_between(thresh_vals, np.zeros(curv_counts.size), curv_counts, color='green', alpha=0.2, label = 'Curvature')

    ax[row][col].plot(thresh_vals, overlap_counts, color='red')
    ax[row][col].fill_between(thresh_vals, np.zeros(overlap_counts.size), overlap_counts, color='red', alpha=0.2, label = 'Intersect')
    ax[row][col].axvline(impact_thresh, linestyle='dashed', color='red')
    ax[row][col].legend()

    #Add secondary axis showing fraction of total number of neurons
    #Following along from: https://matplotlib.org/gallery/api/two_scales.html
    secax = ax[row][col].twinx()
    secax.set_ylabel('Fraction of Recorded Neurons')
    mn, mx = ax[row][col].get_ylim()
    secax.set_ylim(np.true_divide(mn,nn), np.true_divide(mx,nn))
    secax.tick_params(axis="y", labelsize=16)


fig.tight_layout(pad=3, w_pad=4, h_pad=4)
fig.savefig(os.path.join(outputFolder,'highly_weighted.pdf'))

fig3=plt.figure(figsize=[4,4])
plt.plot(rho2_adj_vel, n_impact_vel, '^', markersize=10, color='blue', fillstyle='none', label='Velocity')
plt.plot(rho2_adj_curv, n_impact_curv, 'o', markersize=10, color='green', fillstyle='none', label='Curvature')
plt.plot( np.max((rho2_adj_curv, rho2_adj_vel), axis=0),  n_impact_overlap, 'x', color='red', fillstyle='none',  markersize=10, label='Intersect')
plt.xlabel('Performance ($R^2_{\mathrm{ms},\mathrm{test}}$)')
plt.ylabel('Number of Impactful Neurons')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(0,1)
plt.legend()
fig3.savefig(os.path.join(outputFolder,'nneurons_rho.pdf'))


#Plot the Number of Neurons
import matplotlib.pyplot as plt
import seaborn as sns
fig2=plt.figure(figsize=[2.5,4])
sns.set_style("whitegrid")
axnew = sns.boxplot(data=[n_impact_vel,n_impact_curv, n_impact_overlap])
axnew = sns.swarmplot(data=[n_impact_vel,n_impact_curv, n_impact_overlap], color=".2")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig2.savefig(os.path.join(outputFolder,'number_of_neurons.pdf'))


print(np.median(n_impact_vel), np.median(n_impact_curv), np.median(n_impact_overlap))

print('Median N90 for vel ', np.median(n_impact_vel), np.std(n_impact_vel))
print('Median N90 for curv', np.median(n_impact_curv), np.std(n_impact_curv))
print('Median N90 for intersect', np.median(n_impact_overlap), np.std(n_impact_overlap))



print('Median N90 for vel where rho>0.4', np.median(n_impact_vel[rho2_adj_vel>0.4]), np.std(n_impact_vel[rho2_adj_vel>0.4]))
print('Median N90 for curv where rho>0.4', np.median(n_impact_curv[rho2_adj_curv>0.4]), np.std(n_impact_curv[rho2_adj_curv>0.4]))
print('Median N90 for intersect where rho>0.4', np.median(n_impact_overlap[np.logical_or(rho2_adj_curv>0.4, rho2_adj_vel>0.4)]), np.std(n_impact_overlap[np.logical_or(rho2_adj_curv>0.4, rho2_adj_vel>0.4)]))

print('Median total neurons:', np.median(nnn), ' std:', np.std(nnn))