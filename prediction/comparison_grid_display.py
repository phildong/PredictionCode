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

with open('comparison_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

figtypes = ['bsn', 'bsn_deriv', 'pc', 'pc_deriv', 'slm', 'slm_with_derivs', 'slm_with_derivs_acc', 'slm_tree', 'slm_tree_with_derivs', 'slm_tree_with_derivs_acc']

fig = plt.figure(constrained_layout = True, figsize=(10*len(figtypes), 10*len(keys)+4))
gs = gridspec.GridSpec(len(keys)+1, len(figtypes), figure=fig, width_ratios=tuple(1 for i in range(10)))

for row, key in enumerate(keys):
    for col, figtype in enumerate(figtypes):
        ax = fig.add_subplot(gs[row,col])
        res = data[key][figtype]
        ax.plot(res['time'], res['signal'], 'k', lw=1)
        ax.plot(res['time'], res['output'], 'b', lw=1)
        ax.set_title(figtype+r' $R^2_\mathrm{test}(\mathrm{velocity})$ = %0.3f, $R^2_\mathrm{test}(\mathrm{acceleration})$ = %0.3f' % (res['scorepredicted'], deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx'])))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

ax = fig.add_subplot(gs[-1, :])

score_acc = [[deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx']) for figtype in figtypes] for key in keys]
score_vel = [[data[key][figtype]['scorepredicted'] for figtype in figtypes] for key in keys]
ind = np.arange(len(figtypes))
width = .2

colors = ['red', 'blue', 'green', 'orange']
for i in range(len(keys)):
    autolabel(ax.bar(ind + i*width, score_vel[i], width/2, label=key[-6:], color=colors[i]))
    autolabel(ax.bar(ind + i*width+width/2, score_acc[i], width/2, label=key[-6:], color=colors[i], hatch='/'))

ax.set_xlim(-width/4, len(figtypes)-1+4*width)
ax.set_ylim(0, 1)
ax.grid(axis='y')

fig.savefig('comparison_time_series.png')

fig = plt.figure(constrained_layout = True, figsize=(10*len(figtypes), 10*len(keys)+4))
gs = gridspec.GridSpec(len(keys)+1, len(figtypes), figure=fig, width_ratios=tuple(1 for i in range(10)))

for row, key in enumerate(keys):
    for col, figtype in enumerate(figtypes):
        ax = fig.add_subplot(gs[row,col])
        res = data[key][figtype]
        ax.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go')
        ax.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo')
        ax.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        ax.set_title(figtype+r' $R^2_\mathrm{test}(\mathrm{velocity})$ = %0.3f, $R^2_\mathrm{test}(\mathrm{acceleration})$ = %0.3f' % (res['scorepredicted'], deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx'])))
        ax.set_xlabel('Measured Velocity')
        ax.set_ylabel('Predicted Velocity')

ax = fig.add_subplot(gs[-1, :])

score_acc = [[deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx']) for figtype in figtypes] for key in keys]
score_vel = [[data[key][figtype]['scorepredicted'] for figtype in figtypes] for key in keys]
ind = np.arange(len(figtypes))
width = .2

colors = ['red', 'blue', 'green', 'orange']
for i in range(len(keys)):
    autolabel(ax.bar(ind + i*width, score_vel[i], width/2, label=key[-6:], color=colors[i]))
    autolabel(ax.bar(ind + i*width+width/2, score_acc[i], width/2, label=key[-6:], color=colors[i], hatch='/'))

ax.set_xlim(-width/4, len(figtypes)-1+4*width)
ax.set_ylim(0, 1)
ax.grid(axis='y')

fig.savefig('comparison_scatter.png')