import matplotlib.pyplot as plt
import numpy as np
import pickle

pickled_data = 'new_comparison.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle)

SHOW_GFP = True
if SHOW_GFP:
    pickled_data_GFP = 'new_comparison_aml18.dat'
    with open(pickled_data_GFP, 'rb') as handle:
        dataGFP = pickle.load(handle)

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
delta = .15
offset = .7
ax.set_xticks([0-delta, 1+delta, offset+2-delta, offset+3+delta])
ax.set_xticklabels(['BSN', 'Population', 'BSN (GFP control)', 'Population (GFP control)'])
ax.set_ylabel(r'$R^2_{\mathrm{MS},\mathrm{test}}$', fontsize=20)
ax.set_ylim(-0.5, 1)
ax.set_yticks([ -.5,  0,  .5,  1])
ax.tick_params(axis='both', which='major', labelsize=19)

boxprops = dict(linewidth=2)
capprops = dict(linewidth=2)
whiskerprops = dict(linewidth=2)
medianprops = dict(linewidth=6, color='k')

bsn_rho=np.zeros(len(data.keys()))
slm_rho=np.zeros(len(data.keys()))
for k, key in enumerate(data.keys()): #Comparison line plot
    bsn_rho[k] = data[key]['curvature'][(True, True, True)]['scorespredicted'][1]
    slm_rho[k] = data[key]['curvature'][(True, True, False)]['scorespredicted'][1]
    ax.plot([0, 1], [bsn_rho[k], slm_rho[k]], markersize=5, linewidth=2.5)

ax.boxplot([bsn_rho, slm_rho], positions=[0-delta, 1+delta],
           manage_xticks=False, medianprops=medianprops,
           boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)
print("Total GCAMP datasets:")
print(k+1)

bsn_rho_g = np.zeros(len(dataGFP.keys()))
slm_rho_g = np.zeros(len(dataGFP.keys()))
if SHOW_GFP:
    print("GFP:")
    cmap = plt.cm.get_cmap('gist_earth')
    for k, key in enumerate(dataGFP.keys()):
        bsn_rho_g[k] = dataGFP[key]['curvature'][(True, True, True)]['scorespredicted'][1]
        slm_rho_g[k] = dataGFP[key]['curvature'][(True, True, False)]['scorespredicted'][1]
        thiscolor = cmap(.1+.8*np.true_divide(k,len(dataGFP.keys())))
        if "20210503" in key:
            thiscolor = 'red'
        ax.plot(offset+np.array([2, 3]), [bsn_rho_g[k], slm_rho_g[k]], markersize=5, linewidth=2.5, color=thiscolor)

    ax.boxplot([bsn_rho_g, slm_rho_g], positions=offset+np.array([2-delta, 3+delta]),
               manage_xticks=False, medianprops=medianprops,
               boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)
    print("Total GFP datasets:")
    print(k+1)

#import prediction.provenance as prov
#prov.stamp(ax,.55,.35,__file__)
outfile = 'cms_boxplots_curvature_ica.pdf'
ax.set_title(outfile)
fig.savefig(outfile)
print(outfile)

from scipy.stats import wilcoxon
d = slm_rho - bsn_rho
w, p = wilcoxon(d)
print("P-value for two-sided wilcoxon signed-ranked test on GCAMP")
print(d)
print(p)

from scipy.stats import wilcoxon
d = slm_rho_g - bsn_rho_g
w, p = wilcoxon(d)
print("P-value for two-sided wilcoxon signed-ranked test on GFP")
print(p)

import scipy
print(scipy.version.full_version)

print(" Note I'm not using this version of wilcoxon signed rank tests it uses an approximationt hat is only valid for large numbers")
print(" Instead I use a more recent version that allows me to pass the mode='exact' flag ")

from scipy import stats
t, p = stats.ttest_ind(slm_rho, slm_rho_g, equal_var=False)
print('Welchs unequal variance t-test from population GCAMP to population GFP:')
print(p)

print('mean pop GCAMP:', np.mean(slm_rho), 'mean pop GFP:', np.mean(slm_rho_g))
print('median pop GCAMP:', np.median(slm_rho), 'medain pop GFP:', np.median(slm_rho_g))