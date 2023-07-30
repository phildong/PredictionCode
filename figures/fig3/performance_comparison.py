import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import wilcoxon, ttest_ind

import pickle
import os

from utility import user_tracker

outputFolder = os.path.join(user_tracker.codePath(),'figures/output')
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

for behavior in ['velocity', 'curvature']:
    outfile = '%s/performance_comparison_%s.pdf' % (outputFolder, behavior)

    with open('%s/gcamp_linear_models.dat' % user_tracker.codePath(), 'rb') as handle:
        data = pickle.load(handle)

    with open('%s/gfp_linear_models.dat' % user_tracker.codePath(), 'rb') as handle:
        dataGFP = pickle.load(handle)

    fig, ax = plt.subplots(1, 1, figsize = (15, 15))
    delta = .15
    offset = .7
    ax.set_xticks([0-delta, 1+delta, offset+2-delta, offset+3+delta])
    ax.set_xticklabels(['BSN', 'Population', 'BSN (GFP control)', 'Population (GFP control)'])
    ax.set_ylabel(r'$R^2_{\mathrm{ms}, \mathrm{test}}$', fontsize=20)
    ax.set_ylim(-0.75, 1)
    ax.set_yticks([ -.5,  0,  .5,  1])
    ax.tick_params(axis='both', which='major', labelsize=19)

    boxprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    medianprops = dict(linewidth=6, color='k')

    bsn_rho=np.zeros(len(list(data.keys())))
    slm_rho=np.zeros(len(list(data.keys())))
    for k, key in enumerate(data.keys()): # Comparison line plot
        bsn_rho[k] = data[key][behavior][True]['R2ms_test']
        slm_rho[k] = data[key][behavior][False]['R2ms_test']

        ax.plot([0, 1], [bsn_rho[k], slm_rho[k]], markersize=5, linewidth=2.5)
        
    ax.boxplot([bsn_rho, slm_rho], positions=[0-delta, 1+delta],
            manage_xticks=False, medianprops=medianprops,
            boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)

    print(("Total GCAMP datasets: %d" % len(list(data.keys()))))

    bsn_rho_g = np.zeros(len(list(dataGFP.keys())))
    slm_rho_g = np.zeros(len(list(dataGFP.keys())))
    cmap = plt.cm.get_cmap('gist_earth')
    for k, key in enumerate(dataGFP.keys()):
        bsn_rho_g[k] = dataGFP[key][behavior][True]['R2ms_test']
        slm_rho_g[k] = dataGFP[key][behavior][False]['R2ms_test']
        thiscolor = cmap(.1+.8*np.true_divide(k,len(list(dataGFP.keys()))))
        ax.plot(offset+np.array([2, 3]), [bsn_rho_g[k], slm_rho_g[k]], markersize=5, linewidth=2.5, color=thiscolor)
    
    ax.boxplot([bsn_rho_g, slm_rho_g], positions=offset+np.array([2-delta, 3+delta]),
            manage_xticks=False, medianprops=medianprops,
            boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)
    print(("Total GCAMP datasets: %d" % len(list(dataGFP.keys()))))

    ax.set_title(behavior.capitalize())
    fig.savefig(outfile, backend='pdf', format='pdf')
    print(("Saved: %s" % outfile))

    d = slm_rho - bsn_rho
    w, p = wilcoxon(d)
    print("P-value for two-sided wilcoxon signed-ranked test on GCAMP")
    print(p)

    d = slm_rho_g - bsn_rho_g
    w, p = wilcoxon(d)
    print("P-value for two-sided wilcoxon signed-ranked test on GFP")
    print(p)

    print("NOTE: use more recent version of scipy that supports the mode='exact' flag to get precise Wilcoxon p-values")

    t, p = ttest_ind(slm_rho, slm_rho_g, equal_var=False)
    print('Welchs unequal variance t-test from population GCAMP to population GFP:')
    print(p)

    print(('mean pop GCAMP:', np.mean(slm_rho), 'mean pop GFP:', np.mean(slm_rho_g)))
    print(('median pop GCAMP:', np.median(slm_rho), 'median pop GFP:', np.median(slm_rho_g)))