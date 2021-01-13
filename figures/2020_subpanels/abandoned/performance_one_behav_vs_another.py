import matplotlib.pyplot as plt
import numpy as np
import pickle
outfile = ''

behav0='Curvature'
pickled_data_behav0 = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_curvature_l10.dat'
with open(pickled_data_behav0, 'rb') as handle:
    data0 = pickle.load(handle)

behav1='Velocity'
pickled_data_behav1 = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_velocity_l10.dat'
with open(pickled_data_behav1, 'rb') as handle:
    data1 = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (10, 10), )



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

bsn_rho_all = np.zeros([len(data0), 2])
slm_rho_all = np.zeros([len(data0), 2] )

for i, key in enumerate(data0.keys()):
    bsn_rho = np.array([calc_rho2_adj2(data0, key, 'bsn_deriv'), calc_rho2_adj2(data1, key, 'bsn_deriv')])
    slm_rho = np.array([calc_rho2_adj2(data0, key, 'slm_with_derivs'), calc_rho2_adj2(data1, key, 'slm_with_derivs')])
    ax.plot(slm_rho[0], slm_rho[1], 'o', color='orange', mfc='none', label='population' if i == 0 else "")
    ax.plot(bsn_rho[0], bsn_rho[1], '+', color='blue', label='Best single neuron' if i == 0 else "")
    bsn_rho_all[i, :] = bsn_rho
    slm_rho_all[i, :] = slm_rho

ax.set_aspect('equal')
ax.set_xlabel(behav0)
ax.set_ylabel(behav1)
ax.set_title('rho2 adj 2 performance')
ax.plot([-4,1],[-4,1] , ls="--", c=".3")

import numpy.polynomial.polynomial as poly
coefs_bsn = poly.polyfit(bsn_rho_all[:, 0], bsn_rho_all[:, 1], 1)
x_new_bsn = np.linspace(np.min([bsn_rho_all[:, 0], bsn_rho_all[:, 1]]), np.max([bsn_rho_all[:, 0], bsn_rho_all[:, 1]]), num = 5)
ffit_bsn = poly.polyval(x_new_bsn, coefs_bsn)
plt.plot(x_new_bsn, ffit_bsn, '--', color='blue')
coefs_slm = poly.polyfit(slm_rho_all[:, 0], slm_rho_all[:, 1], 1)
x_new_slm = np.linspace(np.min([slm_rho_all[:, 0], slm_rho_all[:, 1]]), np.max([slm_rho_all[:, 0], slm_rho_all[:, 1]]), num = 5)
ffit_slm = poly.polyval(x_new_slm, coefs_slm)
plt.plot(x_new_slm, ffit_slm, '--', color='orange')

ax.legend()

import prediction.provenance as prov
prov.stamp(ax,.55,.35,__file__)
plt.show()
#fig.savefig(outfile)