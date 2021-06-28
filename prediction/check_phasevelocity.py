import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('neuron_data_bothmc_nb_tmp.dat', 'rb') as handle:
    data = pickle.load(handle)

key = 'BrainScanner20200309_151024'

time = data[key]['time']
raw = data[key]['phase_velocity']

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
fig.suptitle(key, fontsize=24)

ax.plot(time, raw)
ax.set_ylabel('dataHandler PhaseVelocity')

fig.savefig('151024_compare.pdf')