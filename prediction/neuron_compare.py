import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import pickle

with open('neuron_data_bothmc_nb.dat', 'rb') as f:
    data = pickle.load(f)

with open('neuron_data.dat', 'rb') as f:
    data_old = pickle.load(f)

dataset = data['BrainScanner20200130_110803']
dataset_old = data_old['BrainScanner20200130_110803']
time = dataset['time']
neurons = dataset['neurons']
neurons_ica = dataset_old['neurons']

fig, ax = plt.subplots(2, 1, figsize = (10, 10))

ax[0].plot(time, neurons[32,:], label = 'Linear')
ax[0].plot(time, neurons_ica[32,:], label = 'FastICA')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Fluorescence')
ax[0].legend()
ax[0].set_title('AVAR', fontsize = 20)

ax[1].plot(neurons[15,:], label = 'Linear')
ax[1].plot(neurons_ica[15,:], label = 'FastICA')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Fluorescence')
ax[1].legend()
ax[1].set_title('AVAL', fontsize = 20)

print(np.corrcoef(neurons[32,:], neurons_ica[32,:])[0,1])
print(np.corrcoef(neurons[15,:], neurons_ica[15,:])[0,1])

fig.tight_layout()

fig.savefig('neuron_compare.png')