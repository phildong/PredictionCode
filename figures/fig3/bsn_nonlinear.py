import matplotlib.pyplot as plt
import numpy as np
import pickle

from utility import user_tracker

def label_diff(ax, i,js,text,X,Y):
    sjs = sorted(js, key = lambda x: abs(i-x))
    closej = sjs[0]
    x = sum([X[j] for j in sjs])/(len(sjs)+0.)
    # x = (X[i]+X[closej])/2.
    y = max(Y[i], max([Y[j] for j in js]))

    ax.annotate(text, xy=((x+X[i])/2.,1.12*y), zorder=10, ha = 'center', fontsize = 20)
    ax.plot([X[i], X[i], x, x], [1.05*y, 1.1*y, 1.1*y, y], 'k-', linewidth = 2)
    ax.plot(sum([[X[j], X[j], X[j]] for j in sjs],[]), sum([[y, 0.9*y, y] for j in sjs], []), 'k-', linewidth = 2)

def boxplot(ax, x, ys):
    boxprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    medianprops = dict(linewidth=6, color='k')

    ax.boxplot([ys], positions = [x],
           manage_xticks=False, medianprops=medianprops,
           boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)

with open('%s/gcamp_linear_models.dat' % user_tracker.codePath(), 'rb') as f:
    data = pickle.load(f)

with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as f:
    neuron_data = pickle.load(f)

bsns = {}
for k in data.keys():
    vel_idx = np.argmax(np.abs(data[k]['velocity'][True]['weights']))
    curv_idx = np.argmax(np.abs(data[k]['curvature'][True]['weights']))

    _, _, nderiv = rectified_derivative(neuron_data[k]['neurons'])
    neurons_and_derivs = np.vstack((neuron_data[k]['neurons'], nderiv))

    bsns[k] = {'velocity':  {'neuron': neurons_and_derivs[vel_idx,:],  'signal': data[k]['velocity'][True]['signal'],  'time': neuron_data[k]['time'], 'population_fit': data[k]['velocity'][False]['output']},
               'curvature': {'neuron': neurons_and_derivs[curv_idx,:], 'signal': data[k]['curvature'][True]['signal'], 'time': neuron_data[k]['time'], 'population_fit': data[k]['curvature'][False]['output']}}

beh = 'velocity'

fig, ax = plt.subplots(1, 1, figsize = (12, 6))

ax.set_xticks(range(5))
ax.set_xticklabels(['Linear BSN', 'Quadratic BSN', 'Cubic BSN', 'Quartic BSN', 'Population'], fontsize = 16)
ax.set_ylabel(r'$R^2_{\mathrm{MS}}$', fontsize = 20)
ax.set_ylim(0, .75)

cols = np.zeros((5, len(bsns.keys())))
for i, k in enumerate(bsns.keys()):
    neuron = bsns[k][beh]['neuron']
    signal = bsns[k][beh]['signal']

    scores = []
    for order in range(4):
        fit = np.poly1d(np.polyfit(neuron, signal, order + 1))
        scores.append(r2ms(fit(neuron), signal))
        cols[order][i] = scores[-1]
    
    scores.append(r2ms(bsns[k][beh]['population_fit'], signal))
    cols[4][i] = scores[-1]

    ax.plot(scores)

for i in range(5):
    boxplot(ax, i, cols[i,:])

label_diff(ax, 4, [0, 1, 2, 3], "*", range(5), np.max(cols, axis = 1))

fig.savefig('%s/figures/output/bsn_nonlinear.pdf' % user_tracker.codePath())
