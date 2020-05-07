import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.backends.backend_pdf
import userTracker
import os

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3*np.sign(height)),  # 3 points vertical offset
                    fontsize=10,
                    textcoords="offset points",
                    ha='center', va='bottom')

with open('aks_regression_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

# fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# ax.set_ylim(0,1)
# ax.set_ylabel(r'$R^2_\mathrm{test}$', fontsize=16)

# labels = ['Single Model', 'Single Model+Derivative Penalty', 'Shrub', 'Shrub+Derivative Penalty']
# x = np.arange(len(labels))
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=14)
# for key, m in zip(keys, ['o', 'v', '^', 'x', '>', '<', 'D', '*', 'p', 'P', 'd']):
#     slm_r2s = [data[key][i]['R2_test'] for i in [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]]
#     bn_r2s = [data[key][i]['R2_test'] for i in [(0,0,1), (1,0,1), (0,1,1), (1,1,1)]]

#     ax.plot(x, slm_r2s, color='red', marker = m, label=key)
#     ax.plot(x, bn_r2s, color='blue', marker = m)

# ax.legend()
# fig.tight_layout()
# fig.savefig('regression_summary.png')


pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_summary_all.pdf"))

for key in keys:
    results = data[key]

    fig, ax = plt.subplots(3, 2, figsize=(20, 10))
    fig.suptitle(key)

    plots = [(0,0), (0,1), (1,0), (1,1)]

    res = results[plots[0]]
    ax[0,0].plot(res['time'], res['signal'], 'k', lw=1)
    ax[0,0].plot(res['time'], res['output'], 'b', lw=1)
    ax[0,0].set_title(r'Single SLM $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f ($\alpha = %0.2g$, l1_ratio = %0.2g)' % (res['score'], res['scorepredicted'], res['alpha'], res['l1_ratio']))
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Velocity')
    ax[0,0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    res = results[plots[1]]
    ax[1,0].plot(res['time'], res['signal'], 'k', lw=1)
    ax[1,0].plot(res['time'], res['output'], 'b', lw=1)
    ax[1,0].set_title(r'Single Best Neuron $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f' % (res['score'], res['scorepredicted']))
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Velocity')
    ax[1,0].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    ax[2,0].plot(results[plots[0]]['weights'], 'b', label='SLM')
    ax[2,0].plot(results[plots[1]]['weights'], 'g', label='Best Neuron')
    ax[2,0].set_title('Weights')
    ax[2,0].set_xlabel('Neuron')
    ax[2,0].set_ylabel('Weight')
    ax[2,0].legend()

    res = results[plots[2]]
    ax[0,1].plot(res['time'], res['signal'], 'k', lw=1)
    ax[0,1].plot(res['time'], res['output'], 'b', lw=1)
    ax[0,1].set_title(r'Shrub SLM $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f ($\alpha = %0.2g, %0.2g$, l1_ratio = %0.2g, %0.2g)' % (res['score'], res['scorepredicted'], res['res_pos']['alpha'], res['res_neg']['alpha'], res['res_pos']['l1_ratio'], res['res_neg']['l1_ratio']))
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Velocity')
    ax[0,1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    res = results[plots[3]]
    ax[1,1].plot(res['time'], res['signal'], 'k', lw=1)
    ax[1,1].plot(res['time'], res['output'], 'b', lw=1)
    ax[1,1].set_title(r'Shrub Best Neuron $R^2_\mathrm{train}$ = %0.3f, $R^2_\mathrm{test}$ = %0.3f' % (res['score'], res['scorepredicted']))
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].set_ylabel('Velocity')
    ax[1,1].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

    ax[2,1].plot(results[plots[2]]['weights_pos'], 'b', label='SLM+')
    ax[2,1].plot(results[plots[2]]['weights_pos'], 'b', label='SLM-')
    ax[2,1].plot(results[plots[3]]['weights_pos'], 'g', label='Best Neuron+')
    ax[2,1].plot(results[plots[3]]['weights_neg'], 'g', label='Best Neuron-')
    ax[2,1].set_title('Weights')
    ax[2,1].set_xlabel('Neuron')
    ax[2,1].set_ylabel('Weight')
    ax[2,1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)

pdf.close()
