import pickle
import matplotlib.pyplot as plt
import numpy as np

def scores(y, yhat):

    truemean = np.mean(y)
    beta = np.mean(yhat) - truemean
    alpha = np.mean((yhat-truemean)*(y-yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    R2 = 1-np.sum(np.power(y-yhat, 2))/np.sum(np.power(y-truemean, 2))
    rho2 = np.corrcoef(y, yhat)[0,1]**2

    return [R2, rho2 - (alpha+beta**2)**2/((truesigma*predsigma)**2), rho2 - (alpha)**2/((truesigma*predsigma)**2), rho2]

with open('new_model_suite.dat','rb') as f:
    data = pickle.load(f)

for k in data.keys():
    for beh in ['velocity', 'curvature']:
        for mtype in ['tree','mars']:
            m = data[k][beh][mtype]
            data[k][beh][mtype]['scorespredicted'] = scores(m['signal'][m['test_idx']], m['output'][m['test_idx']])

models = data['BrainScanner20200130_110803']['velocity']

plots = ['main', 'no_deriv', 'acc', 'no_deriv_acc', 'l0.01', 'no_deriv_l0.01', 'tree', 'mars']

fig, axs = plt.subplots(4, 2, figsize=(25, 30))

for i, p in enumerate(plots):
    row = i//2
    col = i % 2
    ax = axs[row,col]

    title = p
    m = models[p]

    med = np.mean([data[k]['velocity'][p]['scorespredicted'][1] for k in data.keys()])

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
fig.savefig('new_model_suite.pdf')