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

behavior = 'curvature'
with open('comparison_results_%s_l10.dat' % behavior, 'rb') as handle:
    data = pickle.load(handle)

with open('neuron_data.dat', 'rb') as handle:
    neuron_data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "%s_plots.pdf" % behavior))

for key in keys:

    fig = plt.figure(constrained_layout = True, figsize=(10*len(figtypes), 10*len(figtypes)))
    gs = gridspec.GridSpec(len(figtypes), len(figtypes), figure=fig, width_ratios=[1]*(len(figtypes)))

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


        ts = fig.add_subplot(gs[row, 0])
        ts.plot(res['time'], res['signal'], 'k', lw=1)
        ts.plot(res['time'], res['output'], 'b', lw=1)
        ts.set_xlabel('Time (s)', fontsize = 20, labelpad=20)
        ts.set_ylabel('%s' % behavior, fontsize = 20, labelpad=20)
        ts.set_yticks([])
        ts.tick_params(labelsize=16)
        ts.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

        sc = fig.add_subplot(gs[row, 1])
        sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
        sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
        sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc.text(.1, .7, r'$\rho^2_{\mathrm{adj}} = %0.2f$' % (res['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2), transform = sc.transAxes, fontsize=20)
        sc.set_xlabel('Measured %s' % behavior, fontsize = 20, labelpad=20)
        sc.set_ylabel('Predicted %s' % behavior, fontsize = 20, labelpad=20)
        sc.set_xticks([])
        sc.set_yticks([])
        sc.legend(fontsize = 20)

    fig.suptitle(key)
    fig.tight_layout(rect=[0,.03,1,0.97])

    pdf.savefig(fig)

pdf.close()