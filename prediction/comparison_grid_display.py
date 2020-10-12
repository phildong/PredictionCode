import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import userTracker
import os
from scipy.ndimage import gaussian_filter

def deriv_r2(signal, output, test_idx):
    signal_deriv = gaussian_filter(signal, sigma = 14, order=1)[test_idx]
    output_deriv = gaussian_filter(output, sigma = 14, order=1)[test_idx]

    return 1-np.sum(np.power(signal_deriv - output_deriv, 2))/np.sum(np.power(output_deriv, 2))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0.1:
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='top', color='white')

pickled_data = 'comparison_results_aml18_head_bend_cos_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

#figtypes = ['bsn', 'slm', 'bsn_acc', 'slm_acc', 'bsn_deriv', 'slm_with_derivs', 'bsn_deriv_acc', 'slm_with_derivs_acc', 'pc', 'pc_deriv']
figtypes = ['bsn_deriv', 'slm_with_derivs']

score_R2 = [[0 for figtype in figtypes] for key in keys]
score_rhoadj1 = [[0 for figtype in figtypes] for key in keys]
score_rhoadj2 = [[0 for figtype in figtypes] for key in keys]
score_rho = [[0 for figtype in figtypes] for key in keys]
for i in range(len(keys)):
    for j in range(len(figtypes)):
        res = data[keys[i]][figtypes[j]]
        y = res['signal'][res['test_idx']]
        yhat = res['output'][res['test_idx']]

        truemean = np.mean(y)
        beta = np.mean(yhat) - truemean
        alpha = np.mean((yhat-truemean)*(y-yhat))

        truesigma = np.std(y)
        predsigma = np.std(yhat)

        score_R2[i][j] = 1-np.sum(np.power(y-yhat, 2))/np.sum(np.power(y-truemean, 2))
        score_rhoadj1[i][j] = res['corrpredicted']**2 - (alpha+beta**2)**2/(truesigma*predsigma)**2
        score_rhoadj2[i][j] = res['corrpredicted']**2 - alpha**2/(truesigma*predsigma)**2
        score_rho[i][j] = res['corrpredicted']**2

fig = plt.figure(figsize=(20*len(figtypes), 10*len(keys)+4))
gs = gridspec.GridSpec(len(keys)+1, 2*len(figtypes), figure=fig, width_ratios=tuple(1 for i in range(2*len(figtypes))))

for row, key in enumerate(keys):
    for col, figtype in enumerate(figtypes):
        ax = fig.add_subplot(gs[row,2*col])
        res = data[key][figtype]
        ax.plot(res['time'], res['signal'], 'k', lw=1)
        ax.plot(res['time'], res['output'], 'b', lw=1)
        ax.set_title(figtype+r' scores: (%0.2f, %0.2f, %0.2f, %0.2f)' % (score_R2[row][col], score_rhoadj1[row][col], score_rhoadj2[row][col], score_rho[row][col]))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Curvature')
        ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

        if row == 0:
            ax.text(1.1, 1.2, figtype, va='center', ha='center', fontsize=48, transform=ax.transAxes)
        if col == 0:
            ax.text(-0.5, 0.5, key[12:], va='center', ha='center', fontsize=36, rotation=90, transform=ax.transAxes)

        ax = fig.add_subplot(gs[row,2*col+1])
        res = data[key][figtype]
        ax.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', rasterized = True)
        ax.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', rasterized = True)
        ax.set_title(figtype)
        ax.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        ax.set_xlabel('Measured Curvature')
        ax.set_ylabel('Predicted Curvature')


ax = fig.add_subplot(gs[-1, :])

# score_acc = [[deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx']) for figtype in figtypes] for key in keys]

ind = np.arange(len(figtypes))

colors = ['red', 'blue', 'green', 'orange']
for i in range(len(keys)):
    ax.scatter(ind+.2, score_R2[i])
    ax.scatter(ind+.4, score_rhoadj1[i])
    ax.scatter(ind+.6, score_rhoadj2[i])
    ax.scatter(ind+.8, score_rho[i])
    # autolabel(ax.bar(ind + i*width, score_R2[i], width/4, label=key[-6:], color=colors[i % len(colors)]))
    # autolabel(ax.bar(ind + i*width+width/4, score_rhoadj1[i], width/4, label=key[-6:], color=colors[i % len(colors)]))
    # autolabel(ax.bar(ind + i*width+width/2, score_rhoadj2[i], width/4, label=key[-6:], color=colors[i % len(colors)]))
    # autolabel(ax.bar(ind + i*width+3*width/4, score_rho[i], width/4, label=key[-6:], color=colors[i % len(colors)]))
    # autolabel(ax.bar(ind + i*width+width/2, score_acc[i], width/2, label=key[-6:], color=colors[i], hatch='/'))

for x in ind[1:]:
    ax.axvline(x, linestyle='dashed')

ax.set_xlim(0, len(figtypes))
ax.set_ylim(-1, 1)
ax.set_xticks(np.array([[x+.2, x+.4, x+.6, x+.8] for x in ind]).flatten())
ax.set_xticklabels([r'$R^2$', r'$\rho^2_{\mathrm{adj},1}$', r'$\rho^2_{\mathrm{adj},2}$', r'$\rho^2$']*len(figtypes))
ax.grid(axis='y')

import prediction.provenance as prov
prov.stamp(ax, .55, .15)

print('saving')
# fig.tight_layout()
import os
outfilename = os.path.splitext(os.path.basename(pickled_data))[0] + '.pdf'
fig.savefig(outfilename)
print('saved ' + outfilename )
