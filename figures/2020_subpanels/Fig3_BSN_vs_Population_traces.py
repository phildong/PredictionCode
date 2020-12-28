import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import prediction.userTracker as userTracker
import prediction.provenance as prov
behavior = 'velocity'
#conditions = ['AML18_moving']
conditions = ['AML310_moving']#, 'AML32_moving']
outfilename = 'figures/2020_subpanels/generatedFigs/Fig3_SLM_v_pop_traces_' + behavior + '.pdf'
pickled_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_' + behavior + '_l10.dat'
neural_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/neuron_data.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)
with open(neural_data, 'rb') as handle:
    ndata = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

keys = list(data.keys())
keys.sort()

figtypes = ['bsn_deriv', 'slm_with_derivs']
import os
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), outfilename))

def calc_rho2_adj1(data, key, type='slm_with_derivs'):
    # Calculate rho2adj  (code snippet from comparison_grid_display.py)
    res = data[key][type]
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

def calc_pdf(x, low_lim, high_lim, nbins):
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts))
    return density, bin_centers, bin_edges

def frac_range_covered(true, pred, percentile = True):
    if percentile:
        P = 99
        gap_top = np.max(np.append(np.percentile(true, P)-np.percentile(pred, P), 0))
        gap_bottom = np.max(np.append(np.percentile(pred, 100-P)-np.percentile(true, 100-P), 0))
    else:
        gap_top = np.max(np.append(np.max(true)-np.max(pred), 0))
        gap_bottom = np.max(np.append(np.min(pred)-np.min(true), 0))
    assert gap_bottom >= 0
    assert gap_top >= 0
    range_true = np.ptp(true)
    frac = np.true_divide(range_true - gap_top - gap_bottom, range_true)
    return frac





test_color = u'#2ca02c'
pred_color = u'#1f77b4'
alpha = .5
alpha_test = .3
rho2_adj1 = np.zeros([len(figtypes), len(keys)])
range_covered = np.zeros([len(figtypes), len(keys)])
range_covered_test = np.zeros([len(figtypes), len(keys)])

for i, key in enumerate(keys):
    fig = plt.figure(constrained_layout = True, figsize=[6, 5]) #time series
    gs = gridspec.GridSpec(ncols = 1, nrows = 2, figure = fig)

    fig2 = plt.figure(constrained_layout = True, figsize=[8, 3.2]) #Scatter plot
    gs2 = gridspec.GridSpec(ncols = 2, nrows = 1, figure = fig2)

    fig3 = plt.figure(constrained_layout = True, figsize=[8, 3.2]) #histogram
    gs3 = gridspec.GridSpec(ncols = 2, nrows = 1, figure = fig3)


    ts = [None] * 2
    sc = [None] * 2
    his = [None] * 2
    for model, figtype in enumerate(figtypes):
        rho2_adj1[model, i] = calc_rho2_adj1(data, key, figtype)
        beh = ndata[key][behavior]
        res = data[key][figtype] # the model results are for the z-scored velocity or curvature
        pred = np.sqrt(np.var(beh)) * res['output'] + np.mean(beh) #rescale and offset the results

        range_covered[model, i] = frac_range_covered(beh, pred)
        range_covered_test[model, i] = frac_range_covered(beh[res['test_idx']], pred[res['test_idx']])

        #Plot the time series of the prediction and true
        ts[model] = fig.add_subplot(gs[model, :], sharex = ts[0], sharey = ts[0])
        ts[model].plot(res['time'], beh, 'k', lw = 1.5)
        ts[model].plot(res['time'], pred, color = pred_color, lw = 1.5)
        ts[model].set_xlabel('Time (s)')
        ts[model].set_ylabel(behavior)
        ts[model].fill_between([res['time'][np.min(res['test_idx'])], res['time'][np.max(res['test_idx'])]], np.min(beh), np.max(beh), facecolor = test_color, alpha=.05)
        ts[model].set_xticks(np.arange(0, res['time'][-1], 60))
        ts[model].axhline(linewidth = 0.5, color='k')

        # plot scatter of prediction vs true
        sc[model] = fig2.add_subplot(gs2[0, model], xlabel='Measured '+ behavior, ylabel='Predicted '+ behavior,  sharex = sc[0], sharey = sc[0])
        sc[model].plot(beh[res['train_idx']], pred[res['train_idx']], 'o', color = pred_color, label='Train', rasterized = False, alpha = alpha)
        sc[model].plot(beh[res['test_idx']], pred[res['test_idx']], 'o', color = test_color, label='Test', rasterized = False, alpha = alpha_test)
        sc[model].plot([min(beh), max(beh)], [min(beh), max(beh)], 'k-.')
        sc[model].set_title(figtype + r' $\rho^2_{\mathrm{adj},2}$ = %0.3f' % rho2_adj1[model, i])
        sc[model].legend()
        sc[model].set_aspect('equal', adjustable='box')

        #plot histogram of predicted and true values for held-out test set
        low_lim = -4
        high_lim = 4
        nbins = 24
        pred_hist, pred_bin_centers, pred_bin_edges = calc_pdf(pred[res['test_idx']], low_lim, high_lim, nbins)
        obs_hist, obs_bin_centers, obs_bin_edges = calc_pdf(beh[res['test_idx']], low_lim, high_lim, nbins)
        his[model] = fig3.add_subplot(gs3[0, model], xlabel = behavior, ylabel='Count',
                                      title = figtype + ' frac_range_test = %.2f' % range_covered_test[model, i],
                                      sharex = his[0], sharey = his[0])
        his[model].step(pred_bin_centers, pred_hist, where='mid', label='Prediction')
        his[model].step(obs_bin_centers, obs_hist, where='mid', label='Observation', color='k')
        his[model].set_ylim([0, 1.1*np.max(np.concatenate([pred_hist, obs_hist]))])
        his[model].legend()

    fig.suptitle(key)
    fig2.suptitle(key)
    fig3.suptitle(key)
    #fig.tight_layout(rect=[0,.03,1,0.97])
    pdf.savefig(fig)
    pdf.savefig(fig2)
    pdf.savefig(fig3)

fig4 = plt.figure()
plt.xlabel('Population Performance (Rho2_adj1)')
plt.ylabel('Fraction covered')
plt.plot(rho2_adj1[1, :], range_covered_test[1, :], 'o', label='Population, Test Set')
plt.plot(rho2_adj1[1, :], range_covered_test[0, :], '+', label='BSN, Test Set')
plt.legend()
pdf.savefig(fig4)

fig4andhalf = plt.figure()
plt.xlabel('Population Performance - BSN performance (Rho2_adj1)')
plt.ylabel('Fraction covered')
plt.plot(rho2_adj1[1, :] - rho2_adj1[0, :], range_covered_test[1, :], 'o', label='Population, Test Set')
plt.plot(rho2_adj1[1, :] - rho2_adj1[0, :], range_covered_test[0, :], '+', label='BSN, Test Set')
plt.legend()
pdf.savefig(fig4andhalf)

fig4andthreequarters = plt.figure()
plt.xlabel('Population Performance - BSN performance (Rho2_adj1)')
plt.ylabel('POP Fraction covered - BSN FRaction Covered')
POPminusBSN_rho = rho2_adj1[1, :] - rho2_adj1[0, :]
POPminusBSN_range = range_covered_test[1, :]-range_covered_test[0, :]
plt.plot(POPminusBSN_rho, POPminusBSN_range, 'o')
print("POPminusBSN_range[POPminusBSN_rho>0]")
print(POPminusBSN_range[POPminusBSN_rho>0])
ax475 = plt.gca()
ax475.axvline()
ax475.axhline()
plt.legend()
pdf.savefig(fig4andthreequarters)

fig5 = plt.figure(figsize=[2, 5])
plt.title('Test Only')
plt.ylabel('Fraction Covered')
plt.xlabel('0= BSN, 1 = Pop')
for k in np.arange(len(range_covered_test[1, :])):
    plt.plot(np.array([0, 1]),  np.array([range_covered_test[0, k], range_covered_test[1, k]]))
plt.xticks([0, 1])
pdf.savefig(fig5)


fig6 = plt.figure()
plt.plot(rho2_adj1[1, :], range_covered[1, :], 'o', label='Population, Train + Test')
plt.plot(rho2_adj1[1, :], range_covered[0, :], '+', label='BSN, Train + Test')
plt.legend()
pdf.savefig(fig6)

fig7 = plt.figure(figsize=[2, 5])
plt.title('Train + Test')
plt.ylabel('Fraction Covered')
plt.xlabel('0= BSN, 1 = Pop')
plt.xticks([0,1])
for k in np.arange(len(range_covered[1, :])):
    plt.plot(np.array([0, 1]),  np.array([range_covered[0, k], range_covered[1, k]]))
pdf.savefig(fig7)



pdf.close()
print("wrote "+ outfilename)