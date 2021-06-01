import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
datafolder = '/home/sdempsey/'
outfolder = 'figures/subpanels_revision/generatedFigs/'

def rho_adj(y, yhat):
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))
    beta = np.mean(yhat) - truemean


    truesigma = np.std(y)
    predsigma = np.std(yhat)
    rho2 = np.corrcoef(y, yhat)[0,1]**2
    rho2_adj_1 = (rho2 - (alpha + beta**2) ** 2 / (truesigma * predsigma) ** 2)

    return rho2_adj_1

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

pickled_data = datafolder+'new_comparison.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)#, encoding = 'bytes')

with open(datafolder+'neuron_data_bothmc_nb.dat', 'rb') as f:
    neuron_data = pickle.load(f)#, encoding = 'bytes')

nKeys=11
fig, ax = plt.subplots(3, 4, figsize = (25, 15))
keys = sorted(data.keys(), key = lambda x: -data[x]['velocity'][False]['scorespredicted'][1])
rho2_adj_vel = np.array([data[keys[x]]['velocity'][False]['scorespredicted'][1] for x in np.arange(nKeys)])
rho2_adj_curv = np.array([data[keys[x]]['curvature'][False]['scorespredicted'][1] for x in np.arange(nKeys)])

#preallocate arrays for plotting later
n_impact_vel, n_impact_curv,  n_impact_overlap= np.zeros(nKeys), np.zeros(nKeys), np.zeros(nKeys)

nnn=np.zeros(nKeys)
for ii, dataset in enumerate(keys[:nKeys]):
    print(dataset)
    nnn[ii] = neuron_data[dataset]['neurons'].shape[0]
    row = ii // 4
    col = ii % 4

    vel_res = data[dataset]['velocity'][False]
    curv_res = data[dataset]['curvature'][False]

    neurons_unn = neuron_data[dataset][b'neurons']
    _, _, nderiv = rectified_derivative(neurons_unn)
    neurons_and_derivs = np.vstack((neurons_unn, nderiv))

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

    vel_rhos = np.array([rho_adj(vel_res[b'signal'], vel_lines[i,:]) for i in range(vel_order.size)])
    curv_rhos = np.array([rho_adj(curv_res[b'signal'], curv_lines[i,:]) for i in range(curv_order.size)])

    ax[row][col].set_title(dataset[12:] + ' N=%d' %nn)
    ax[row][col].set_xlim((0, 1))
    ax[row][col].set_ylabel('Number of Neurons')
    ax[row][col].set_xlabel(r'Fraction of full model performance')
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
import os
import prediction.provenance as prov
import userTracker as userTracker
outpath = os.path.join(userTracker.codePath(), outfolder)
fig.savefig(os.path.join(outpath,'highly_weighted.pdf'), metadata=prov.pdf_metadata(__file__))

fig3=plt.figure(figsize=[4,4])
plt.plot(rho2_adj_vel, n_impact_vel, '^', markersize=10, color='blue', fillstyle='none', label='Velocity')
plt.plot(rho2_adj_curv, n_impact_curv, 'o', markersize=10, color='green', fillstyle='none', label='Curvature')
plt.plot( np.max((rho2_adj_curv, rho2_adj_vel), axis=0),  n_impact_overlap, 'x', color='red', fillstyle='none',  markersize=10, label='Intersect')
plt.xlabel('Performance (rho2_adj_1)')
plt.ylabel('Number of Impactful Neurons')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(0,1)
plt.legend()
fig3.savefig(os.path.join(outpath,'nneurons_rho.pdf'), metadata=prov.pdf_metadata(__file__))


#Plot the Number of NEurons
import matplotlib.pyplot as plt
import seaborn as sns
fig2=plt.figure(figsize=[2.5,4])
sns.set_style("whitegrid")
axnew = sns.boxplot(data=[n_impact_vel,n_impact_curv, n_impact_overlap])
axnew = sns.swarmplot(data=[n_impact_vel,n_impact_curv, n_impact_overlap], color=".2")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig2.savefig(os.path.join(outpath,'number_of_neurons.pdf'), metadata=prov.pdf_metadata(__file__))


print(np.median(n_impact_vel), np.median(n_impact_curv), np.median(n_impact_overlap))

print('Median N90 for vel ', np.mean(n_impact_vel), np.std(n_impact_vel))
print('Median N90 for curv', np.mean(n_impact_curv), np.std(n_impact_curv))
print('Median N90 for intersect', np.mean(n_impact_overlap), np.std(n_impact_overlap))



print('Median N90 for vel where rho>0.4', np.mean(n_impact_vel[rho2_adj_vel>0.4]), np.std(n_impact_vel[rho2_adj_vel>0.4]))
print('Median N90 for curv where rho>0.4', np.mean(n_impact_curv[rho2_adj_curv>0.4]), np.std(n_impact_curv[rho2_adj_curv>0.4]))
print('Median N90 for intersect where rho>0.4', np.mean(n_impact_overlap[np.logical_or(rho2_adj_curv>0.4, rho2_adj_vel>0.4)]), np.std(n_impact_overlap[np.logical_or(rho2_adj_curv>0.4, rho2_adj_vel>0.4)]))

print('Median total neurons:', np.median(nnn), ' std:', np.std(nnn))