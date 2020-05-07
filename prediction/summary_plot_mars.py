import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.backends.backend_pdf
import userTracker
import os

with open('mars_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()


pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_summary_mars.pdf"))

for key in keys:
    res = data[key]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle(key)

    ax.plot(res['time'], res['signal'], 'k', lw=1)
    ax.plot(res['time'], res['output'], 'b', lw=1)
    ax.set_title(r'MARS $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f' % (res['score'], res['scorepredicted']))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)

pdf.close()
