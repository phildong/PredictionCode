import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('comparison_results_velocity_l10.dat', 'rb') as handle:
    vel = pickle.load(handle)

with open('comparison_results_curvature_l10.dat', 'rb') as handle:
    curv = pickle.load(handle)

fig, axs = plt.subplots(2, 2, figsize = (20, 20))

axs[0,0].set_xlabel(r'Velocity $\rho^2_{\mathrm{adj},1}$', fontsize = 16)
axs[0,0].set_ylabel(r'Velocity $\rho^2_{\mathrm{adj},2}$', fontsize = 16)

axs[1,0].set_xlabel(r'Velocity $\rho^2_{\mathrm{adj},1}$', fontsize = 16)
axs[1,0].set_ylabel(r'Velocity $R^2$', fontsize = 16)

axs[0,1].set_xlabel(r'Curvature $\rho^2_{\mathrm{adj},1}$', fontsize = 16)
axs[0,1].set_ylabel(r'Curvature $\rho^2_{\mathrm{adj},2}$', fontsize = 16)

axs[1,1].set_xlabel(r'Curvature $\rho^2_{\mathrm{adj},1}$', fontsize = 16)
axs[1,1].set_ylabel(r'Curvature $R^2$', fontsize = 16)

for row in axs:
    for ax in row:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axvline(0, linestyle='dashed')
        ax.axhline(0, linestyle='dashed')
        ax.plot([-1, 1], [-1, 1], linestyle='dashed')

velscores0 = []
velscores1 = []
velscores2 = []
curvscores0 = []
curvscores1 = []
curvscores2 = []

for key in curv.keys():
    velscores0.append(vel[key]['slm_with_derivs']['scorespredicted'][0])
    curvscores0.append(curv[key]['slm_with_derivs']['scorespredicted'][0])
    velscores1.append(vel[key]['slm_with_derivs']['scorespredicted'][1])
    curvscores1.append(curv[key]['slm_with_derivs']['scorespredicted'][1])
    velscores2.append(vel[key]['slm_with_derivs']['scorespredicted'][2])
    curvscores2.append(curv[key]['slm_with_derivs']['scorespredicted'][2])

    axs[0,0].text(velscores1[-1], velscores2[-1], key[12:])
    axs[0,1].text(curvscores1[-1], curvscores2[-1], key[12:])
    axs[1,0].text(velscores1[-1], velscores0[-1], key[12:])
    axs[1,1].text(curvscores1[-1], curvscores0[-1], key[12:])

axs[0,0].scatter(velscores1, velscores2)
axs[0,1].scatter(curvscores1, curvscores2)
axs[1,0].scatter(velscores1, velscores0)
axs[1,1].scatter(curvscores1, curvscores0)

fig.savefig('metric_comparison.pdf')