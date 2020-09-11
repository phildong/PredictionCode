import matplotlib.pyplot as plt
import numpy as np
import pickle
outfile = 'performance_comparison_deriv_rho2_l10.pdf'

pickled_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/comparison_results_velocity_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

pickled_data_GFP = '/projects/LEIFER/PanNeuronal/decoding_analysis/comparison_results_aml18_l10.dat'
with open(pickled_data_GFP, 'rb') as handle:
    dataGFP = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['BSNd', 'SLMd', 'BSNd (GFP control)', 'SLMd (GFP control)'], fontsize=16)
ax.set_ylabel(r'$\rho^2_{\mathrm{adj},2}$', fontsize=16)
ax.set_ylim(-2, 1)

def calc_rho2_adj2(data, key, type='slm_with_derivs'):
    # Calculate rho2adj  (code snippet from comparison_grid_display.py)
    res = data[key][type]
    y = res['signal'][res['test_idx']]
    yhat = res['output'][res['test_idx']]

    truemean = np.mean(y)
    beta = np.mean(yhat) - truemean
    alpha = np.mean((yhat - truemean) * (y - yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    return (res['corrpredicted'] ** 2 - alpha ** 2 / (truesigma * predsigma) ** 2)

for key in data.keys():
    bsn_rho = calc_rho2_adj2(data, key, 'bsn_deriv')
    slm_rho = calc_rho2_adj2(data, key, 'slm_with_derivs')
    ax.plot([0, 1], [bsn_rho, slm_rho], markersize=5)

ax.axvline(1.5, linestyle='dashed')

for key in dataGFP.keys():
    bsn_rho = calc_rho2_adj2(dataGFP, key, 'bsn_deriv')
    slm_rho = calc_rho2_adj2(dataGFP, key, 'slm_with_derivs')
    ax.plot([2, 3], [bsn_rho, slm_rho], markersize=5, color='k')


import prediction.provenance as prov
prov.stamp(ax,.55,.35,__file__)
ax.set_title(outfile)
fig.savefig(outfile)