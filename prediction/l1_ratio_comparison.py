import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('comparison_results_curvature.dat', 'rb') as handle:
    data = pickle.load(handle)

with open('comparison_results_curvature_cv.dat', 'rb') as handle:
    dataCV = pickle.load(handle)

with open('comparison_results_curvature_l10.dat', 'rb') as handle:
    datal10 = pickle.load(handle)

fig, ax = plt.subplots(2, 1, figsize = (10, 10), sharex = True)

ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(['l1_ratio = 0', 'cross-validated', 'l1_ratio = 0.03'], fontsize=16)

ax[0].set_ylabel(r'$\rho^2_{\mathrm{adj},2}$', fontsize=8)
ax[0].set_ylim(0, 1)

for key in datal10.keys():
    res_slm = data[key]['slm_with_derivs']
    res_slm_cv = dataCV[key]['slm_with_derivs']
    res_slm_l10 = datal10[key]['slm_with_derivs']

    ax[0].plot([0, 1, 2], [res_slm_l10['scorespredicted'][2], res_slm_cv['scorespredicted'][2], res_slm['scorespredicted'][2]], markersize=5)

ax[1].set_ylabel(r'Number of neurons above mean magnitude', fontsize=8)
ax[1].set_ylim(0, 125)

for key in dataCV.keys():
    res_slm = data[key]['slm_with_derivs']
    res_slm_cv = dataCV[key]['slm_with_derivs']
    res_slm_l10 = datal10[key]['slm_with_derivs']

    nn = lambda xs: len(np.where(np.abs(xs) > np.mean(np.abs(xs)))[0])

    ax[1].plot([0, 1, 2], [nn(res_slm_l10['weights']), nn(res_slm_cv['weights']), nn(res_slm['weights'])], markersize=5)

fig.savefig('l1_ratio_comparison_curvature.pdf')