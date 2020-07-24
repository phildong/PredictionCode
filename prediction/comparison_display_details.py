import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import userTracker
import os
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

def deriv_r2(signal, output, test_idx):
    signal_deriv = gaussian_filter(signal, sigma = 14, order=1)[test_idx]
    output_deriv = gaussian_filter(output, sigma = 14, order=1)[test_idx]

    return 1-np.sum(np.power(signal_deriv - output_deriv, 2))/np.sum(np.power(output_deriv, 2))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0.1:
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='top', color='white')

with open('comparison_results.dat', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "slm_deriv_results.pdf"))

for key in keys:

    fig = plt.figure(constrained_layout = True, figsize=(5*(len(figtypes)+1), 20))
    gs = gridspec.GridSpec(len(figtypes)+1, 2, figure=fig, width_ratios=(1, 1))

    for row, figtype in enumerate(figtypes):
        res = data[key][figtype]


        y = res['signal'][res['test_idx']]
        yhat = res['output'][res['test_idx']]

        truemean = np.mean(y)
        beta = np.mean(yhat) - truemean
        alpha = np.mean((yhat-truemean)*(y-yhat))

        truesigma = np.std(y)
        predsigma = np.std(yhat)
        R2 = 1-np.sum(np.power(y-yhat, 2))/np.sum(np.power(y-truemean, 2))

        print(beta**2, alpha)
        print(res['corrpredicted']**2, (beta/truesigma)**2, ((alpha+beta**2)**2/(truesigma*predsigma)**2))
        print("Actual R^2:  ",R2)
        print("Formula R^2: ",res['corrpredicted']**2 - (beta/truesigma)**2 - (alpha+beta**2)**2/((truesigma*predsigma)**2))


        ts = fig.add_subplot(gs[row+1, 0])
        ts.plot(res['time'], res['signal'], 'k', lw=1)
        ts.plot(res['time'], res['output'], 'b', lw=1)
        # w = np.ones(res['time'].size, dtype=bool)
        # w[:6] = 0
        # w[-6:] = 0
        # ts.fill_betweenx(res['output'], np.roll(res['time'], -6), np.roll(res['time'], 6), where=w, color='b', lw=1)
        ts.set_title(figtype+r' $R^2_\mathrm{test}(\mathrm{velocity})$ = %0.3f, $R^2_\mathrm{test}(\mathrm{acceleration})$ = %0.3f' % (R2, deriv_r2(data[key][figtype]['signal'], data[key][figtype]['output'], data[key][figtype]['test_idx'])))
        ts.set_xlabel('Time (s)')
        ts.set_ylabel('Velocity')
        ts.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

        sc = fig.add_subplot(gs[row+1, 1])
        sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
        sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
        sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc.set_title(figtype+r' $\rho^2_{\mathrm{test},\mathrm{adj}}(\mathrm{velocity})$ = %0.3f' % (res['corrpredicted']**2 - (alpha+beta**2)**2/((truesigma*predsigma)**2)))
        sc.set_xlabel('Measured Velocity')
        sc.set_ylabel('Predicted Velocity')
        sc.legend()

    ax = fig.add_subplot(gs[0, :])

    slm_weights = data[key]['slm_with_derivs']['weights']
    bsn_weights = data[key]['bsn_deriv']['weights']

    n = slm_weights.size/2

    xs = np.argsort(slm_weights[:n])

    scaler = MinMaxScaler()
    scaler.fit(np.array([slm_weights[xs]]).T)

    ax.plot(scaler.transform(np.array([slm_weights[xs]]).T)+4, 'k.', label = 'SLM')
    ax.plot(scaler.transform(np.zeros(xs.size).reshape(1,-1)).T+4, 'b', linestyle='dashed')

    # scaler.fit(np.array([slm_weights[xs+n]]).T)

    ax.plot(scaler.transform(np.array([slm_weights[xs+n]]).T)+2, 'k.')
    ax.plot(scaler.transform(np.zeros(xs.size).reshape(1,-1)).T+2, 'b', linestyle='dashed')

    # scaler.fit(np.array([slm_weights[xs+2*n]]).T)

    # ax.plot(scaler.transform(np.array([slm_weights[xs+2*n]]).T), 'k.')
    # ax.plot(scaler.transform(np.zeros(xs.size).reshape(1,-1)).T, 'b', linestyle='dashed')

    for i in range(2):
        bn = np.array([bsn_weights[xs+i*n]]).T
        bn_idx = np.nonzero(bn)[0]
        if bn_idx.size > 0:
            # scaler.fit(np.array([slm_weights[xs+i*n]]).T)
            ax.bar(bn_idx[0], scaler.transform(bn)[bn_idx[0]] - scaler.transform([[0]])[0,0], bottom = scaler.transform([[0]])[0,0]+4-2*i, color='g', label = 'Best Neuron')

    ax.axhline(1.5)
    ax.axhline(3.5)
    ax.set_ylim(1.5, 5.5)

    ax.text(-2, 3.8, '$F$', fontsize=16)
    ax.text(-2, 1.8, r'$\left.\frac{dF}{dt}\right.$', fontsize=16)
    # ax.text(-2, -0.2, r'$\left.\frac{dF}{dt}\right|_-$', fontsize=16)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(xs)
    ax.set_yticks([])
    ax.legend()

    fig.suptitle(key)
    fig.tight_layout(rect=[0,.03,1,0.97])

    pdf.savefig(fig)

pdf.close()