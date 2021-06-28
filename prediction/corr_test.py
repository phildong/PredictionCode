import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
datafolder = ''

pickled_data = datafolder+'new_comparison.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)#, encoding = 'bytes')

with open(datafolder+'neuron_data_bothmc_nb.dat', 'rb') as f:
    neuron_data = pickle.load(f)#, encoding = 'bytes')

fig, ax = plt.subplots(1, 2, figsize = (25, 10))

ax[0].set_xlabel('Weight', fontsize=20)
ax[1].set_xlabel('Weight', fontsize=20)

ax[0].set_ylabel(r'$\rho_{\mathrm{all}}$', fontsize = 20)
ax[1].set_ylabel(r'$\rho_{\mathrm{train}}$', fontsize = 20)

ax[0].set_ylim(-.7, .7)
ax[1].set_ylim(-.7, .7)

weights = data['BrainScanner20200130_110803']['velocity'][False]['weights']
train_idx = data['BrainScanner20200130_110803']['velocity'][False]['train_idx']

neurons = neuron_data['BrainScanner20200130_110803']['neurons']
signal = neuron_data['BrainScanner20200130_110803']['cmsvelocity']

corrs_all = [np.corrcoef(n, signal)[0,1] for n in neurons]
corrs_train = [np.corrcoef(n[train_idx], signal[train_idx])[0,1] for n in neurons]

ax[0].plot(weights[:len(corrs_all)], corrs_all, 'ko', alpha = 0.5)
ax[1].plot(weights[:len(corrs_all)], corrs_train, 'ko', alpha = 0.5)

ax[0].set_title(r'$\rho = %0.2f$' % np.corrcoef(weights[:len(corrs_all)], corrs_all)[0,1], fontsize = 24)
ax[1].set_title(r'$\rho = %0.2f$' % np.corrcoef(weights[:len(corrs_all)], corrs_train)[0,1], fontsize = 24)

fig.savefig('corr_test.pdf')
