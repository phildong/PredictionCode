import pickle
import matplotlib.pyplot as plt
import numpy as np

from utility import user_tracker

with open('%s/all_trained_models.dat' % user_tracker.codePath(),'rb') as f:
    data = pickle.load(f)

models = data['BrainScanner20200130_110803']['velocity']

plots = ['main', 'no_deriv', 'acc', 'no_deriv_acc', 'l0.01', 'no_deriv_l0.01', 'tree', 'mars']

fig, axs = plt.subplots(4, 2, figsize=(25, 30))

for i, p in enumerate(plots):
    row = i//2
    col = i % 2
    ax = axs[row,col]

    title = p
    m = models[p]

    med = np.mean([data[k]['velocity'][p]['R2ms_test'] for k in data.keys()])

    ax.plot(m['time'], m['signal'], 'k', lw = 1)
    ax.plot(m['time'], m['output'], 'b', lw = 1)
    ax.text(-.05, 1.07, chr(i+97)+'.', fontsize=50, transform=ax.transAxes)
    ax.set_title(r'$R^2_\mathrm{ms, test} = %0.2f$, mean: %0.2f' % (m['R2ms_test'], med), fontsize = 36, pad = 32)
    ax.set_xlabel('Time (s)', fontsize = 30)
    ax.set_ylabel('Velocity', fontsize = 30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.fill_between([m['time'][np.min(m['test_idx'])], m['time'][np.max(m['test_idx'])]], min(np.min(m['output']), np.min(m['signal'])), max(np.max(m['output']), np.max(m['signal'])), facecolor='gray', alpha = 0.5)

fig.tight_layout()
fig.savefig('%s/figures/output/model_comparison.pdf' % user_tracker.codePath())