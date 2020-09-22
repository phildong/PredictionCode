import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('comparison_results.dat', 'rb') as handle:
    vel = pickle.load(handle)

with open('comparison_results_curvature.dat', 'rb') as handle:
    curv = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

ax.set_xlabel(r'Velocity $\rho^2_{\mathrm{adj},2}$', fontsize=16)
ax.set_ylabel(r'Curvature $\rho^2_{\mathrm{adj},2}$', fontsize=16)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axvline(0, linestyle='dashed')
ax.axhline(0, linestyle='dashed')

velscores = []
curvscores = []

for key in curv.keys():
    velscores.append(vel[key]['slm_with_derivs']['scorespredicted'][2])
    curvscores.append(curv[key]['slm_with_derivs']['scorespredicted'][2])

    ax.text(velscores[-1], curvscores[-1], key[12:])

ax.scatter(velscores, curvscores)

fig.savefig('vel_vs_curv_comparison.pdf')