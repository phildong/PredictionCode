import matplotlib.pyplot as plt
import numpy as np
import pickle

behavior = 'velocity'
outfile = 'performance_comparison_deriv_rho2_'+behavior+'_l10.pdf'

pickled_data = 'comparison_results_'+behavior+'_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

SHOW_GFP = True
if SHOW_GFP:
    pickled_data_GFP = 'comparison_results_aml18_'+behavior+'_l10.dat'
    with open(pickled_data_GFP, 'rb') as handle:
        dataGFP = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['BSNd', 'SLMd', 'BSNd (GFP control)', 'SLMd (GFP control)'], fontsize=16)
ax.set_ylabel(r'$\rho^2_{\mathrm{adj}}$', fontsize=16)
ax.set_ylim(-0.8, 1)

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
    rho2_adj = (res['corrpredicted'] ** 2 - (alpha + beta**2) ** 2 / (truesigma * predsigma) ** 2)
    print(key + ': %.2f' % rho2_adj)
    return rho2_adj

bsn_rho=np.zeros(len(data.keys()))
slm_rho=np.zeros(len(data.keys()))
for k, key in enumerate(data.keys()): #Comparison line plot
    bsn_rho[k] = calc_rho2_adj2(data, key, 'bsn_deriv')
    slm_rho[k] = calc_rho2_adj2(data, key, 'slm_with_derivs')
    ax.plot([0, 1], [bsn_rho[k], slm_rho[k]], markersize=5)
ax.boxplot([bsn_rho, slm_rho], positions=[0, 1], manage_xticks=False, medianprops=dict(linewidth=4))


bsn_rho_g = np.zeros(len(data.keys()))
slm_rho_g = np.zeros(len(data.keys()))
if SHOW_GFP:
    print("GFP:")
    for k, key in enumerate(dataGFP.keys()):
        bsn_rho_g[k] = calc_rho2_adj2(dataGFP, key, 'bsn_deriv')
        slm_rho_g[k] = calc_rho2_adj2(dataGFP, key, 'slm_with_derivs')
        ax.plot([2, 3], [bsn_rho_g[k], slm_rho_g[k]], markersize=5, color='k')
    ax.boxplot([bsn_rho_g, slm_rho_g], positions=[2, 3], manage_xticks=False, medianprops=dict(linewidth=4))

# import prediction.provenance as prov
# prov.stamp(ax,.55,.35,__file__)
# ax.set_title(outfile)
fig.savefig(outfile)