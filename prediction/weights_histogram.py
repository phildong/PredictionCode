import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('comparison_results_velocity_l10.dat', 'rb') as handle:
    data_vel = pickle.load(handle)

with open('comparison_results_curvature_l10.dat', 'rb') as handle:
    data_curv = pickle.load(handle)

fig, ax = plt.subplots(3, 4, figsize=(25, 15))
keys = sorted(data_vel.keys(), key = lambda x: -data_vel[x]['slm_with_derivs']['scorespredicted'][2])

for i, key in enumerate(keys):
    row = i // 4
    col = i % 4

    weights_vel = data_vel[key]['slm_with_derivs']['weights']
    weights_curv = data_curv[key]['slm_with_derivs']['weights']

    bins = np.linspace(-.1, .1, 30)
    vel_hist, _ = np.histogram(weights_vel, bins = bins)
    curv_hist, _ = np.histogram(weights_curv, bins = bins)

    ax[row][col].fill_between(.5*(bins[1:] + bins[:-1]), np.zeros(vel_hist.shape), vel_hist, color='blue', alpha=0.5, label = 'Velocity Weights')
    ax[row][col].fill_between(.5*(bins[1:] + bins[:-1]), np.zeros(curv_hist.shape), curv_hist, color='green', alpha=0.5, label = 'Curvature Weights')

    ax[row][col].set_title(key[-6:]+": (%0.2f, %0.2f)" % (data_vel[key]['slm_with_derivs']['scorespredicted'][2], data_curv[key]['slm_with_derivs']['scorespredicted'][2]))
    ax[row][col].legend()

fig.tight_layout(pad=3, w_pad=4, h_pad=4)
plt.show()
