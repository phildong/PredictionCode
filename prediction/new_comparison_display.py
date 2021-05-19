import SLM
from Classifier import rectified_derivative
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import userTracker

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "new_comparison_grid.pdf"))

with open('new_comparison.dat', 'rb') as f:
    results = pickle.load(f)

for key in results.keys():
    fig, ax = plt.subplots(2, 2, figsize = (25, 15))
    fig.suptitle(key, fontsize = 36)
    for behi, beh in zip([0, 1], ['velocity', 'curvature']):
        for bsn in [True, False]:
            row = behi
            res = results[key][beh][bsn]

            # ax[row][0].text(-0.1, 0.5, label, rotation = 90, fontsize = 24, verticalalignment = 'center', transform=ax[row][0].transAxes)

            ax[row][0+bsn].set_ylabel('CMS Velocity' if behi == 0 else 'Geometric Curvature', fontsize=16)
            ax[row][0+bsn].plot(res['time'], res['signal']*res['signal_std'] + res['signal_mean'], 'k', lw=1)
            ax[row][0+bsn].plot(res['time'], res['output']*res['signal_std'] + res['signal_mean'], 'b', lw=1)
            ax[row][0+bsn].set_xlabel('Time (s)', fontsize=16)
            ax[row][0+bsn].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']*res['signal_std'] + res['signal_mean']), np.max(res['signal']*res['signal_std'] + res['signal_mean']), facecolor='gray', alpha = 0.5)
            ax[row][0+bsn].set_title(('BSN ' if bsn else 'Population ')+r'$R^2_\mathrm{ms} = %0.3f$' % res['scorespredicted'][1], fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(left = 0.05, top = 0.9)
    pdf.savefig(fig)

pdf.close()