import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.backends.backend_pdf
import userTracker
import os

with open('tree_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "tree_summary_all.pdf"))

for key in keys:
    results = data[key]

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(key)

    res = results['velocity']
    ax[0].plot(res['time'], res['signal'], 'k', lw=1)
    ax[0].plot(res['time'], res['output'], 'b', lw=1)
    ax[0].set_title(r'Velocity Decision Tree $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f)' % (res['score'], res['scorepredicted']))
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Velocity')
    ax[0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    res = results['curvature']
    ax[1].plot(res['time'], res['signal'], 'k', lw=1)
    ax[1].plot(res['time'], res['output'], 'b', lw=1)
    ax[1].set_title(r'Curvature Decision Tree $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f)' % (res['score'], res['scorepredicted']))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Curvature')
    ax[1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)

pdf.close()
