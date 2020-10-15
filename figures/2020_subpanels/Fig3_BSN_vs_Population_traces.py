import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import prediction.userTracker as userTracker
import prediction.dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
import prediction.provenance as prov

#conditions = ['AML18_moving']
conditions = ['AKS297.51_moving']#, 'AML32_moving']
outfilename = 'figures/2020_subpanels/generatedFigs/Fig3_SLM_v_pop_traces.pdf'
behavior = 'velocity'
pickled_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_' + behavior + '_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}



keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']

import os

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), outfilename))

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


rho2_adj2 = np.zeros([len(figtypes), len(keys)])

test_color = u'#2ca02c'
pred_color = u'#1f77b4'
alpha = .5
alpha_test = .3

for i, key in enumerate(keys):
    fig = plt.figure(constrained_layout=True, figsize=[6, 5])
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

    fig2 = plt.figure(constrained_layout=True, figsize=[8, 4])
    gs2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)


    ts = [None] * 2
    sc = [None] * 2
    for model, figtype in enumerate(figtypes):
        rho2_adj2[model, i] = calc_rho2_adj2(data, key, figtype)
        res = data[key][figtype]

        ts[model] = fig.add_subplot(gs[model, :], sharex=ts[0], sharey=ts[0])
        ts[model].plot(res['time'], res['signal'], 'k', lw=1.5)
        ts[model].plot(res['time'], res['output'], color=pred_color, lw=1.5)
        ts[model].set_xlabel('Time (s)')
        ts[model].set_ylabel('Velocity')
        ts[model].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor=test_color, alpha=.05)
        ts[model].set_xticks(np.arange(0, res['time'][-1], 60))
        ts[model].axhline(linewidth=0.5, color='k')

        sc[model] = fig2.add_subplot(gs2[0, model], xlabel='Measured Velocity', ylabel='Predicted Velocity',  sharex=sc[0], sharey=sc[0])
        sc[model].plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'o', color=pred_color, label='Train', rasterized=False, alpha=alpha)
        sc[model].plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'o', color=test_color, label='Test', rasterized=False, alpha=alpha_test)
        sc[model].plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc[model].set_title(figtype + r' $\rho^2_{\mathrm{adj},2}(\mathrm{velocity})$ = %0.3f' % rho2_adj2[model, i])
        sc[model].legend()
        sc[model].set_aspect('equal', adjustable='box')

    fig.suptitle(key)
    fig2.suptitle(key)
    #fig.tight_layout(rect=[0,.03,1,0.97])
    pdf.savefig(fig)
    pdf.savefig(fig2)


pdf.close()
print("wrote "+ outfilename)