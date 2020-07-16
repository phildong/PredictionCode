import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('comparison_results.dat', 'rb') as handle:
    data = pickle.load(handle)

with open('comparison_results_aml18.dat', 'rb') as handle:
    dataGFP = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['BSN', 'SLM', 'BSN (GFP control)', 'SLM (GFP control)'], fontsize=16)
ax.set_ylabel(r'$\rho^2_{\mathrm{adj},12}$', fontsize=16)

ax.set_ylim(-1, 1)

for key in data.keys():
    res_bsn = data[key]['bsn']
    y = res_bsn['signal'][res_bsn['test_idx']]
    yhat = res_bsn['output'][res_bsn['test_idx']]
    
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))
    beta = np.mean(y-yhat)

    truesigma = np.std(y)
    predsigma = np.std(yhat)

    bsn_rho = res_bsn['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2

    res_slm = data[key]['slm']
    y = res_slm['signal'][res_slm['test_idx']]
    yhat = res_slm['output'][res_slm['test_idx']]
    
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))
    beta = np.mean(y-yhat)

    truesigma = np.std(y)
    predsigma = np.std(yhat)

    slm_rho = res_slm['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2

    ax.plot([0, 1], [bsn_rho, slm_rho], markersize=5)

ax.axvline(1.5, linestyle='dashed')

for key in dataGFP.keys():
    res_bsn = dataGFP[key]['bsn']
    y = res_bsn['signal'][res_bsn['test_idx']]
    yhat = res_bsn['output'][res_bsn['test_idx']]
    
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))
    beta = np.mean(y-yhat)

    truesigma = np.std(y)
    predsigma = np.std(yhat)

    bsn_rho = res_bsn['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2

    res_slm = dataGFP[key]['slm']
    y = res_slm['signal'][res_slm['test_idx']]
    yhat = res_slm['output'][res_slm['test_idx']]
    
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))
    beta = np.mean(y-yhat)

    truesigma = np.std(y)
    predsigma = np.std(yhat)

    slm_rho = res_slm['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2

    ax.plot([2, 3], [bsn_rho, slm_rho], markersize=5, color='k')

fig.savefig('performance_comparison_rho1.pdf')