import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import userTracker
import dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

with open('comparison_results_curvature_l10.dat', 'rb') as handle:
    data = pickle.load(handle)

with open('neuron_data.dat', 'rb') as handle:
    neuron_data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "slm_deriv_curvature_l10_weight_results.pdf"))

for key in keys:

    fig = plt.figure(constrained_layout = True, figsize=(10*(len(figtypes)+2), 10*len(figtypes)))
    gs = gridspec.GridSpec(len(figtypes), len(figtypes)+2, figure=fig, width_ratios=[1]*(len(figtypes)+2))

    for row, figtype in enumerate(figtypes):
        res = data[key][figtype]


        y = res['signal'][res['test_idx']]
        yhat = res['output'][res['test_idx']]

        truemean = np.mean(y)
        beta = np.mean(yhat) - truemean
        alpha = np.mean((yhat-truemean)*(y-yhat))

        truesigma = np.std(y)
        predsigma = np.std(yhat)
        R2 = 1-np.sum(np.power(y-yhat, 2))/np.sum(np.power(y-truemean, 2))

        print(beta**2, alpha)
        print(res['corrpredicted']**2, (beta/truesigma)**2, ((alpha+beta**2)**2/(truesigma*predsigma)**2))
        print("Actual R^2:  ",R2)
        print("Formula R^2: ",res['corrpredicted']**2 - (beta/truesigma)**2 - (alpha+beta**2)**2/((truesigma*predsigma)**2))


        ts = fig.add_subplot(gs[row, 0])
        ts.plot(res['time'], res['signal'], 'k', lw=1)
        ts.plot(res['time'], res['output'], 'b', lw=1)
        # w = np.ones(res['time'].size, dtype=bool)
        # w[:6] = 0
        # w[-6:] = 0
        # ts.fill_betweenx(res['output'], np.roll(res['time'], -6), np.roll(res['time'], 6), where=w, color='b', lw=1)
        ts.set_xlabel('Time (s)')
        ts.set_ylabel('Velocity')
        ts.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

        sc = fig.add_subplot(gs[row, 1])
        sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
        sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
        sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc.set_title(figtype+r' $\rho^2_{\mathrm{adj},2}$ = %0.3f' % (res['corrpredicted']**2 - (alpha)**2/((truesigma*predsigma)**2)))
        sc.set_xlabel('Measured Velocity')
        sc.set_ylabel('Predicted Velocity')
        sc.legend()

    ax = fig.add_subplot(gs[:, 2:])

    slm_weights_raw = data[key]['slm_with_derivs']['weights'][:data[key]['slm_with_derivs']['weights'].size/2]
    correlations = [np.corrcoef(x, data[key]['slm_with_derivs']['signal'])[0,1] for x in neuron_data[key]['neurons']]

    ax.scatter(correlations, slm_weights_raw)
    ax.axvline(0, linestyle='dashed')
    ax.axhline(0, linestyle='dashed')
    ax.set_xlabel(r'$\rho$', fontsize=14)
    ax.set_ylabel('Weight', fontsize=14)

    fig.suptitle(key)
    fig.tight_layout(rect=[0,.03,1,0.97])

    pdf.savefig(fig)

pdf.close()