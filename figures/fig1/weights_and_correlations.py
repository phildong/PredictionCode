import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import gridspec

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

import pickle
import os

from utility import user_tracker
from utility import data_handler as dh

def calc_pdf(x, low_lim, high_lim, nbins):
    bin_width = np.true_divide(high_lim-low_lim, nbins)
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts)*bin_width)
    return density, bin_centers, bin_edges

def compare_pdf(a, b, low_lim=-3, high_lim=3, nbins=24, alabel="", blabel="", PDF=None, suplabel=""):
    a_hist, a_bin_centers, a_bin_edges = calc_pdf(a, low_lim, high_lim, nbins)
    b_hist, bin_centers, bin_edges = calc_pdf(b, low_lim, high_lim, nbins)
    assert np.all(a_bin_edges==bin_edges)

    hfig = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(2, 1, figure=hfig)
    ha = hfig.add_subplot(gs[0, 0])
    ha.step(bin_centers, a_hist, where='mid', label=alabel, lw=4)
    ha.step(bin_centers, b_hist, where='mid', label=blabel, lw=4)
    ha.axvline(0, color="black")
    ha.axvline(np.nanmean(a), linestyle='dashed', color='blue', label='mean ' + alabel)
    ha.axvline(np.nanmean(b), linestyle='dashed', color='orange', label='mean ' + blabel)

    #Add two gaussians, each with the variance  of a or b
    x = np.linspace(bin_centers[0],bin_centers[-1], 100)
    ha.plot(x, norm.pdf(x, scale=np.nanstd(a)),
            'r-', lw=3, alpha=0.6, color='blue',
            label=alabel + ' gaussian sigma = %.3f' % np.nanstd(a))
    ha.plot(x,  norm.pdf(x, scale=np.nanstd(b)),
            'r-', lw=3, alpha=0.6, color='orange',
            label=blabel + ' gaussian sigma = %.3f' % np.nanstd(b))
    ha.legend()
    ha.yaxis.tick_right()
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ha.set_xlim(low_lim, high_lim)
    max_yticks = 3
    yloc = plt.MaxNLocator(max_yticks)
    ha.yaxis.set_major_locator(yloc)
    ha.spines["left"].set_visible(False)
    ha.spines["right"].set_visible(True)
    ha.tick_params(labelsize=17)

    hb = hfig.add_subplot(gs[1, 0])
    hb.step(bin_centers, a_hist - b_hist, where='mid', label="A-B")
    hb.axvline(0, color="black")
    hb.axhline()

    ha.set_title(alabel + " " + blabel)
    hb.set_title('Residual: ' + alabel + " - " + blabel)

    ylim_high = np.max([a_hist, b_hist])
    ylim_low = np.min(a_hist-b_hist)
    hb.set_xlim(low_lim, high_lim)
    hb.set_ylim(ylim_low, ylim_high)

    ha.set_ylabel('Probability Density')
    hb.set_ylabel('Probability Density')

    MSE = np.sum((a_hist - b_hist)**2)/a_hist.size
    hfig.suptitle(suplabel + ' MSE = %.4f ' % MSE)
    if PDF is not None:
        pdf.savefig(hfig)
    return MSE

outputFolder = os.path.join(user_tracker.codePath(), 'figures/output')
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

with open('%s/gcamp_linear_models.dat' % user_tracker.codePath(), 'rb') as handle:
    data = pickle.load(handle)
with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as handle:
    neuron_data = pickle.load(handle)

for behavior in ['velocity', 'curvature']:
    keys = sorted(list(data.keys()))

    outfilename = behavior + '_weights.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, outfilename))

    Frac_dFdt = np.zeros(len(keys))
    R2ms = np.zeros((2, len(keys)))

    for i, key in enumerate(keys):
        fig = plt.figure(constrained_layout=True, figsize=(10*(2), 10*2))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1]*(2))

        for bsn in [True, False]:
            res = data[key][behavior][bsn]
            R2ms[1-bsn, i] = res['R2ms_test']

            ts = fig.add_subplot(gs[1-bsn, 0])
            ts.plot(res['time'], res['signal'], 'k', lw=1)
            ts.plot(res['time'], res['output'], 'b', lw=1)
            ts.set_xlabel('Time (s)')
            ts.set_ylabel(behavior.capitalize())
            ts.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

            sc = fig.add_subplot(gs[1-bsn, 1], xlabel='Measured %s' % behavior.capitalize(), ylabel='Predicted %s' % behavior.capitalize())
            sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
            sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
            sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
            sc.set_title(('BSN' if bsn else 'Population') +r' $R^2_{\mathrm{ms},\mathrm{test}}(\mathrm{%s})$ = %0.3f' % (behavior, R2ms[1-bsn, i]))
            sc.legend()

        fig2 = plt.figure(figsize=[7,7])
        ax2 = fig2.add_subplot(111, xlabel=r'$\rho$', ylabel='Weight')

        nneurons = data[key][behavior][bsn]['weights'].size/2
        slm_weights_raw = data[key][behavior][bsn]['weights'][:nneurons]
        slm_weights_raw_deriv = data[key][behavior][bsn]['weights'][nneurons:]
        correlations = [np.corrcoef(x, data[key][behavior][bsn]['signal'])[0,1] for x in neuron_data[key]['neurons']]
        deriv_correlations = [np.corrcoef(x, data[key][behavior][bsn]['signal'])[0,1] for x in neuron_data[key]['neuron_derivatives']]

        Frac_dFdt[i] = np.sum(np.abs(slm_weights_raw_deriv)) /  (np.sum( np.abs(slm_weights_raw)) + np.sum(np.abs(slm_weights_raw_deriv) ))

        ax2.plot(correlations, slm_weights_raw, 'o', label='F',  markersize=8 )
        ax2.plot(deriv_correlations, slm_weights_raw_deriv, 'o', markersize=8, color='orange', label='dF/dt')

        if key == 'BrainScanner20200130_110803':
            AVAR = 32
            AVAL = 15
            ax2.text(correlations[AVAR], slm_weights_raw[AVAR], 'AVAR')
            ax2.text(deriv_correlations[AVAR], slm_weights_raw_deriv[AVAR], 'AVAR')
            ax2.text(correlations[AVAL], slm_weights_raw[AVAL], 'AVAL')
            ax2.text(deriv_correlations[AVAL], slm_weights_raw_deriv[AVAL], 'AVAL')
            
        rho = np.concatenate((np.array(correlations), np.array(deriv_correlations)), axis=None)
        weights = np.concatenate((np.array(slm_weights_raw), np.array(slm_weights_raw_deriv)), axis=None)
        coefs = np.polyfit(rho, weights, 1)
        x_new = np.linspace(np.min(rho), np.max(rho), num=3)
        ffit = np.polyval(coefs, x_new)
        plt.plot(x_new, ffit,'r--')

        ax2.axvline(0, color="black")
        ax2.axhline(0, color="black")
        ax2.set_title(key + ',  %.2f Percent of Weights come from derivatives' % Frac_dFdt[i])
        ax2.legend()

        win_scalar=1.05
        largest_weight = np.nanmax(np.abs([slm_weights_raw, slm_weights_raw_deriv]))
        ax2.set_ylim(-largest_weight*win_scalar, largest_weight*win_scalar)
        ax2.spines["top"].set_visible(True)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(labelsize=17)
        pdf.savefig(fig2)

        fig.suptitle(key)
        fig.tight_layout(rect=[0,.03,1,0.97])
        pdf.savefig(fig)

        # Plot distribution of weights
        compare_pdf(slm_weights_raw, slm_weights_raw_deriv,
                    low_lim=-largest_weight, high_lim=largest_weight, nbins=24,
                    alabel='F', blabel='dF/dt', PDF=pdf, suplabel='PDF of population decoder weights\n' + key)
        # Scatterplot of F weights vs dF/dt weights
        f = plt.figure()
        h = f.add_subplot(1,1,1, xlabel='weight of F', ylabel='weight of dF/dt', title='Weights of each neuron\n' + key)
        h.scatter(slm_weights_raw, slm_weights_raw_deriv, color='black')
        h.axhline(color='black')
        h.axvline(color='black')
        h.set_xlim(-largest_weight*win_scalar, largest_weight*win_scalar)
        h.set_ylim(-largest_weight*win_scalar, largest_weight*win_scalar)
        pdf.savefig(f)

    figsummary = plt.figure()
    axs = figsummary.add_subplot(1, 1, 1, xlabel='Decoder Performance (rho2_adj1)',
                                ylabel='Percentage of magnitude of weights allocated to F',
                                title='Balance of weights allocated to F vs dF/dt for population decoder')
    axs.scatter(R2ms[1, :], 1 - Frac_dFdt)
    axs.axhline(0.5)
    axs.set_ylim(0, 1)
    pdf.savefig(figsummary)

    pdf.close()
    print(("wrote "+ outfilename))