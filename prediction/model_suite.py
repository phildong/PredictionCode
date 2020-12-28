import pickle
import matplotlib.pyplot as plt
import numpy as np

data = dict()
elasticdata = dict()

for i in range(11):
    with open('comparison_results_all_models_velocity_l10_%d.dat' % i,'rb') as f:
        data.update(pickle.load(f))

    with open('comparison_results_all_models_velocity_l10.01_%d.dat' % i,'rb') as f:
        elasticdata.update(pickle.load(f))

models = data['BrainScanner20200130_110803']
elasticmodels = elasticdata['BrainScanner20200130_110803']

plots = [(False, 'slm_with_derivs', 'LM + Derivs'),
         (False, 'slm', 'LM'), 
         (False, 'slm_with_derivs_acc', 'LM + Derivs + Acc'), 
         (False, 'slm_acc', 'LM + Acc'), 
         (True, 'slm_with_derivs', 'Elastic LM + Derivs'),
         (True, 'slm', 'Elastic LM'),  
         (False, 'mars', 'MARS + Derivs'),
         (False, 'slm_tree_with_derivs', 'LM + Derivs + Tree')]

fig, axs = plt.subplots(4, 2, figsize=(25, 30))

for i, p in enumerate(plots):
    row = i//2
    col = i % 2
    ax = axs[row,col]

    el, mtype, title = p
    m = elasticmodels[mtype] if el else models[mtype]

    med = np.mean([(elasticdata if el else data)[k][mtype]['scorespredicted'][1] for k in data.keys()])

    ax.plot(m['time'], m['signal'], 'k', lw = 1)
    ax.plot(m['time'], m['output'], 'b', lw = 1)
    ax.text(-.05, 1.07, chr(i+97)+'.', fontsize=50, transform=ax.transAxes)
    ax.set_title(r'$R^2_\mathrm{ms, test} = %0.2f$, mean: %0.2f' % (m['scorespredicted'][1], med), fontsize = 36, pad = 32)
    ax.set_xlabel('Time (s)', fontsize = 30)
    ax.set_ylabel('Velocity', fontsize = 30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.fill_between([m['time'][np.min(m['test_idx'])], m['time'][np.max(m['test_idx'])]], min(np.min(m['output']), np.min(m['signal'])), max(np.max(m['output']), np.max(m['signal'])), facecolor='gray', alpha = 0.5)

fig.tight_layout()
fig.savefig('model_suite.pdf')