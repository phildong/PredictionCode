import matplotlib.pyplot as plt
import numpy as np
import pickle

fig, ax = plt.subplots(1, 1, figsize = (20, 10))

with open('neuron_data.dat', 'rb') as f:
    neuron_data = pickle.load(f)
    keyList = np.sort(neuron_data.keys())

for i in range(11):
    with open('decimation_velocity_%d.dat' % i, 'rb') as handle:
        data = pickle.load(handle)

        xs = [k for k in data.keys()]
        ys = np.array([data[k] for k in data.keys()])


        if i == 5:
            ax.plot(xs, ys, label = keyList[i][21:], color='k')
        else:
            ax.plot(xs, ys, label = keyList[i][21:])

ax.set_xlabel('Number of Neurons', fontsize = 16)
ax.set_ylabel(r'$\rho^2_{\mathrm{adj},2}$', fontsize = 16)
ax.set_ylim(0, 1)

ax.legend()

fig.savefig('decimation_velocity.pdf')