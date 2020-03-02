import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.backends.backend_pdf
import userTracker
import os

def shade_wrong(ax, time, correct, color, ymin, ymax):
    state = 1
    start = 0
    for i in range(correct.size):
        if state == 1 and not correct[i]:
            start = i
            state = 0
        elif state == 0 and correct[i]:
            state = 1
            ax.fill_between([time[start], time[i]], ymin, ymax, color=color)

with open('clf_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "clf_summary_all.pdf"))

for key in keys:
    results = data[key]    

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(key)

    res = results['velocity']
    correct = res['output'] == np.sign(res['signal'])

    ax[0].plot(res['time'], res['signal'], 'k', lw=1)
    shade_wrong(ax[0], res['time'], correct, 'red', np.min(res['signal']), np.max(res['signal']))
    ax[0].axhline(0)
    ax[0].set_title(r'Velocity Classifier $\mathrm{score}_\mathrm{train}$ = %0.3f, $\mathrm{score}_\mathrm{test}$ = %0.3f' % (res['score'], res['scorepredicted']))
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Velocity')
    ax[0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    res = results['curvature']
    correct = res['output'] == np.sign(res['signal'])

    ax[1].plot(res['time'], res['signal'], 'k', lw=1)
    shade_wrong(ax[1], res['time'], correct, 'red', np.min(res['signal']), np.max(res['signal']))
    ax[1].axhline(0)
    ax[1].set_title(r'Curvature Classifier $\mathrm{score}_\mathrm{train}$ = %0.3f, $\mathrm{score}_\mathrm{test}$ = %0.3f' % (res['score'], res['scorepredicted']))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Curvature')
    ax[1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)

pdf.close()
