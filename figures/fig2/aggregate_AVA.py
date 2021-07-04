import os
import numpy as np

from scipy import stats

from utility import user_tracker
from utility import dataHandler as dh

import matplotlib.pyplot as plt

with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as f:
    neuron_data = pickle.load(f)

outputFolder = os.path.join(user_tracker.codePath(),'figures/output')

### AVA indices
idns = ['BrainScanner20200130_110803', 'BrainScanner20200130_110803', 'BrainScanner20200130_105254', 'BrainScanner20200310_141211', 'BrainScanner20200310_141211', 'BrainScanner20200310_142022', 'BrainScanner20200310_142022'       ]
neurons = [32,                          15,                           95,                             71,                           42,                            15,                                     16        ]

z_activity = np.array([])
z_activity_deriv = np.array([])
vel_bucket = np.array([])

for idn, neuron in zip(idns, neurons):
    activity = neuron_data[idn]['neurons']
    activity_deriv = neuron_data[idn]['neuron_derivatives']
    vel = neuron_data[idn]['velocity']

    z_activity = np.append(z_activity, stats.zscore(activity[neuron]))
    z_activity_deriv = np.append(z_activity, stats.zscore(activity[neuron]))
    vel_bucket = np.append(vel_bucket, vel)

print("plotting aggregate figure")

# Calculate bins for box plot and split data up into subarrays based on bin
nbins = 13
plus_epsilon = 1.00001
bin_edges = np.linspace(np.nanmin(vel_bucket) * plus_epsilon, np.nanmax(vel_bucket) * plus_epsilon, nbins)
binwidth = np.diff(bin_edges)
assigned_bin = np.digitize(vel_bucket, bin_edges)
activity_bin = [None] * (len(bin_edges) - 1)
deriv_bin = [None] * (len(bin_edges) - 1)
for k, each in enumerate(np.unique(assigned_bin)):
    idx = np.argwhere(assigned_bin == each)[:, 0]
    activity_bin[k] = z_activity[idx]
    deriv_bin[k] = z_activity_deriv[idx]

fig, ax = plt.subplots(2, 1, figsize=(4, 8))
ax[0].plot(vel_bucket, z_activity, '.', color=u'#1f77b4', alpha=.05, zorder=10)
ax[1].plot(vel_bucket, z_activity_deriv, '.', color=u'#ff7f0e', alpha=.05, zorder=10)

boxprops = dict(linewidth=.5)
capprops = dict(linewidth=.5)
whiskerprops = dict(linewidth=.5)
flierprops = dict(linewidth=.2, markersize=1, marker='+')
medianprops = dict(linewidth=2, color='k')#'#67eb34')
labels = [''] * len(activity_bin)

for i, bins in enumerate(activity_bin, deriv_bin):
    ax[i].boxplot(bins, positions=bin_edges[:-1] + binwidth / 2, widths=binwidth * .9, boxprops=boxprops,
               medianprops=medianprops, labels=labels, manage_xticks=False,
               capprops=capprops, whiskerprops=whiskerprops, flierprops=flierprops, zorder=20)
    ax[i].locator_params(nbins=5)
    ax[i].axhline(color='k', linewidth=.5)
    ax[i].xlim([-.2, 0.3])
    ax[i].xlabel('velocity (mm^-1 s)')
    ax[i].ylabel('AVA activity (z-score(' + type + '))')
    ax[i].legend()

fig.savefig('%s/figures/output/aggregate_AVA.pdf' % user_tracker.codePath())