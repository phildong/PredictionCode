import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import userTracker
import dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
import prediction.provenance as prov

#conditions = ['AML18_moving']
outfolder = 'figures/subpanels_revision/generatedFigs/'
import userTracker as userTracker


conditions = ['AKS297.51_moving', 'AML32_moving']
behavior = 'velocity' #'velocity' #'curvature'
pickled_data = '/home/sdempsey/new_comparison.dat' #'/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_' + behavior + '_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

def take_deriv(neurons):
    from prediction.Classifier import rectified_derivative
    _, _, nderiv = rectified_derivative(neurons)
    return nderiv

def calc_pdf(x, low_lim, high_lim, nbins):
    bin_width = np.true_divide(high_lim-low_lim, nbins)
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts)*bin_width)
    return density, bin_centers, bin_edges

def compare_pdf(a, b, low_lim=-3, high_lim=3, nbins=24, alabel="", blabel="", PDF=None, suplabel=""):
    a_hist, a_bin_centers, a_bin_edges = calc_pdf(a, low_lim, high_lim, nbins)
    b_hist, bin_centers, bin_edges = calc_pdf(b, low_lim, high_lim, nbins)
    assert np.all(a_bin_edges==bin_edges), 'Andy screwed up the code.'

    hfig = plt.figure(figsize=[10,10])
    gs = gridspec.GridSpec(2, 1, figure=hfig)
    ha = hfig.add_subplot(gs[0, 0])
    ha.step(bin_centers, a_hist, where='mid', label=alabel, lw=4)
    ha.step(bin_centers, b_hist, where='mid', label=blabel, lw=4)
    ha.axvline(0, color="black")
    ha.axvline(np.nanmean(a), linestyle='dashed', color='blue', label='mean ' + alabel)
    ha.axvline(np.nanmean(b), linestyle='dashed', color='orange', label='mean ' + blabel)

    #Add two gaussians, each with the variance  of a or b
    from scipy.stats import norm
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

    prov.stamp(hb, .55, .35, __file__ + '\n'+ pickled_data)


    MSE = np.sum((a_hist - b_hist)**2)/a_hist.size
    hfig.suptitle(suplabel + ' MSE = %.4f ' % MSE)
    if PDF is not None:
        pdf.savefig(hfig)
    return MSE

neuron_data = {}
deriv_neuron_data = {}
for typ_cond in conditions:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))


    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
    keyList = np.sort(dataSets.keys())

    for key in keyList:
        if key in excludeSets:
            continue
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']

        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
        
        neuron_data[key] = neurons
        deriv_neuron_data[key] = take_deriv(neurons)

keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']

import os
outfilename = os.path.splitext(os.path.basename(pickled_data))[0] + '_'+ behavior + '_weights.pdf'

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), outfilename), metadata=prov.pdf_metadata(__file__))

def  type_helper(in_type):
    if in_type is 'slm_with_derivs':
        return False
    elif in_type is 'bsn_deriv':
        return True
    else:
        assert False
    return

def calc_rho2_adj(data, key, behavior, type='slm_with_derivs'):
    type = type_helper(type)

    # Calculate rho2adj  (code snippet from com ison_grid_display.py)
    res = data[key][behavior][type]
    y = res['signal'][res['test_idx']]
    yhat = res['output'][res['test_idx']]

    truemean = np.mean(y)
    beta = np.mean(yhat) - truemean
    alpha = np.mean((yhat - truemean) * (y - yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    rho2_adj = (res['corrpredicted'] ** 2 - (alpha + beta**2) ** 2 / (truesigma * predsigma) ** 2)
    print(key + ': %.2f' % rho2_adj)
    return rho2_adj



Frac_dFdt = np.zeros(len(keys))
rho2_adj1 = np.zeros([len(figtypes), len(keys)])
for i, key in enumerate(keys):

    fig = plt.figure(constrained_layout=True, figsize=(10*(len(figtypes)+2), 10*len(figtypes)))
    gs = gridspec.GridSpec(len(figtypes), len(figtypes)+2, figure=fig, width_ratios=[1]*(len(figtypes)+2))

    for row, figtype in enumerate(figtypes):
        rho2_adj1[row, i] = calc_rho2_adj(data, key, behavior, figtype)
        res = data[key][behavior][type_helper(figtype)]

        ts = fig.add_subplot(gs[row, 0])
        ts.plot(res['time'], res['signal'], 'k', lw=1)
        ts.plot(res['time'], res['output'], 'b', lw=1)
        # w = np.ones(res['time'].size, dtype=bool)
        # w[:6] = 0
        # w[-6:] = 0
        # ts.fill_betweenx(res['output'], np.roll(res['time'], -6), np.roll(res['time'], 6), where=w, color='b', lw=1)
        ts.set_xlabel('Time (s)')
        ts.set_ylabel('Velocity')
        ts.fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(res['signal']), np.max(res['signal']), facecolor='gray', alpha = 0.5)

        sc = fig.add_subplot(gs[row, 1], xlabel='Measured Velocity', ylabel='Predicted Velocity')
        sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
        sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
        sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc.set_title(figtype +r' $\rho^2_{\mathrm{adj},2}(\mathrm{velocity})$ = %0.3f' % rho2_adj1[row, i])
        sc.legend()

    fig2 = plt.figure(figsize=[7,7])
    ax2 = fig2.add_subplot(111, xlabel=r'$\rho$', ylabel='Weight')

    slm_weights_raw = data[key][behavior][type_helper('slm_with_derivs')]['weights'][:data[key][behavior][type_helper('slm_with_derivs')]['weights'].size/2]
    slm_weights_raw_deriv = data[key][behavior][type_helper('slm_with_derivs')]['weights'][data[key][behavior][type_helper('slm_with_derivs')]['weights'].size/2:]
    correlations = [np.corrcoef(x, data[key][behavior][type_helper('slm_with_derivs')]['signal'])[0,1] for x in neuron_data[key]]
    deriv_correlations = [np.corrcoef(x, data[key][behavior][type_helper('slm_with_derivs')]['signal'])[0,1] for x in deriv_neuron_data[key]]

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
    import numpy.polynomial.polynomial as poly
    try:
        rho = np.concatenate((np.array(correlations), np.array(deriv_correlations)), axis=None)
        weights = np.concatenate((np.array(slm_weights_raw), np.array(slm_weights_raw_deriv)), axis=None)
        coefs = poly.polyfit(rho, weights, 1)
        x_new = np.linspace(np.min(rho), np.max(rho), num=3)
        ffit = poly.polyval(x_new, coefs)
        plt.plot(x_new, ffit,'r--')
    except:
        None

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
    #prov.stamp(ax2, .55, .35, __file__ + '\n'+ pickled_data)
    pdf.savefig(fig)

    # Plot distribution of weights
    compare_pdf(slm_weights_raw, slm_weights_raw_deriv,
                low_lim=-largest_weight, high_lim=largest_weight, nbins=24,
                alabel='F', blabel='dF/dt', PDF=pdf, suplabel='PDF of population decoder weights\n' + key)
    # Scatterplot of F weights vs dF/dt weights
    f =plt.figure()
    h = f.add_subplot(1,1,1, xlabel='weight of F', ylabel='weight of dF/dt', title='Weights of each neuron\n' + key)
    h.scatter(slm_weights_raw, slm_weights_raw_deriv, color='black')
    h.axhline(color='black')
    h.axvline(color='black')
    h.set_xlim(-largest_weight*win_scalar, largest_weight*win_scalar)
    h.set_ylim(-largest_weight*win_scalar, largest_weight*win_scalar)
    #prov.stamp(ax2, .55, .35, __file__ + '\n' + pickled_data)
    pdf.savefig(f)


figsummary = plt.figure()
axs = figsummary.add_subplot(1, 1, 1, xlabel='Decoder Performance (rho2_adj1)',
                             ylabel='Percentage of magnitude of weights allocated to F',
                             title='Balance of weights allocated to F vs dF/dt for population decoder')
axs.scatter(rho2_adj1[1, :], 1 - Frac_dFdt)
axs.axhline(0.5)
axs.set_ylim(0, 1)
prov.stamp(axs, .55, .35, __file__ + '\n'+ pickled_data)
pdf.savefig(figsummary)

pdf.close()
print("wrote "+ outfilename)