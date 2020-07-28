import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
import userTracker
import dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

with open('/projects/LEIFER/PanNeuronal/decoding_analysis/comparison_results.dat', 'rb') as handle:
    data = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

neuron_data = {}
mean_intensity = []
percentile_intensity = []
frac_nan = []
rho2 = []
recording_length = []
label = []
R_mean = []
R_mean_fano_factor = []
G_mean = []
G_mean_fano_factor = []
mean_vel = []
vel_std = []
std_test_train_ratio = []
std_vel_test = []
vel_pdf_mse = []
vel_pdf_kl =  []

def calc_pdf(x, low_lim, high_lim, nbins):
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts))
    return density, bin_centers, bin_edges

from scipy.special import kl_div
def compare_pdf(a, b, low_lim=-3, high_lim=3, nbins=24, alabel="", blabel="", PDF=None, suplabel=""):
    a_hist, a_bin_centers, a_bin_edges = calc_pdf(a, low_lim, high_lim, nbins)
    b_hist, bin_centers, bin_edges = calc_pdf(b, low_lim, high_lim, nbins)
    assert np.all(a_bin_edges==bin_edges), 'Andy screwed up the code.'

    hfig = plt.figure()
    gs = gridspec.GridSpec(2, 1, figure=hfig)
    ha = hfig.add_subplot(gs[0, 0])
    ha.step(bin_centers, a_hist, where='mid', label=alabel)
    ha.step(bin_centers, b_hist, where='mid', label=blabel)
    ha.legend()
    hb = hfig.add_subplot(gs[1, 0])
    hb.step(bin_centers, a_hist - b_hist, where='mid', label="A-B")
    hb.axhline()

    ha.set_title(alabel + " " + blabel)
    hb.set_title('Residual: ' + alabel + " - " + blabel)

    ylim_high = np.max([a_hist, b_hist])
    ylim_low = np.min(a_hist-b_hist)
    ha.set_ylim(ylim_low, ylim_high)
    hb.set_ylim(ylim_low, ylim_high)

    ha.set_ylabel('Probability Density')
    hb.set_ylabel('Probability Density')

    MSE = np.sum((a_hist - b_hist)**2)/a_hist.size
    #KL = kl_div(a_hist, b_hist)



    hfig.suptitle(suplabel + ' MSE = %.4f ' % MSE)

    if PDF is not None:
        pdf.savefig(hfig)
    return MSE#, KL

filename = 'correlates_of_performance.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

for typ_cond in ['AKS297.51_moving', 'AML32_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # data parameters
    dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
            'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5,  # gauss window for red and green channel
            'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
            'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
            }
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
    keyList = np.sort(dataSets.keys())

    for key in keyList:
        if key in excludeSets:
            continue
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_raw = dataSets[key]['Neurons']['I_smooth']
        R = dataSets[key]['Neurons']['R']
        G = dataSets[key]['Neurons']['G']
        vel = dataSets[key]['BehaviorFull']['AngleVelocity']
        res = data[key]['slm_with_derivs']
        vel_train = res['signal'][res['train_idx']]
        vel_test = res['signal'][res['test_idx']]




        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
        
        neuron_data[key] = neurons


        mean_intensity.append(np.nanmean(neurons))
        rho2.append(data[key]['slm_with_derivs']['corrpredicted']**2)
        percentile_intensity.append(np.nanpercentile(neurons, 75))
        frac_nan.append(np.true_divide(np.sum(np.sum(np.isnan(neurons_raw))), neurons_raw.size))
        recording_length.append(neurons.shape[1])
        R_mean.append(np.nanmean(R))
        R_mean_fano_factor.append(np.nanmedian((np.nanstd(G, 1)**2 / np.nanmean(G, 1) )) )
        G_mean.append(np.nanmean(R))
        G_mean_fano_factor.append(np.nanmedian((np.nanstd(G, 1)**2 / np.nanmean(G, 1) )) )
        mean_vel.append(np.nanmean(vel))
        vel_std.append(np.nanstd(vel))
        std_test_train_ratio.append(np.nanstd(vel_test) / np.nanstd(vel_train))
        std_vel_test.append(np.nanstd(vel_test))
        mse = compare_pdf(vel_test, vel_train, alabel="test", blabel="train", suplabel=key + "rho2 = %.2f " % rho2[-1], PDF=pdf)
        vel_pdf_mse.append(mse)
        #vel_pdf_kl.append(kl)

        label.append(key)




import matplotlib.backends.backend_pdf

def plot_candidate(x,  x_name, metric  = 'rho2', metric_name = 'rho2', labels=label, PDF=None):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, rho2)
    ax.set_xlabel(x_name)
    ax.set_ylabel(metric_name)
    ax.set_xlim(np.min(x)*.9, np.max(x)*1.4)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], rho2[i]))

    import prediction.provenance as prov
    prov.stamp(ax,.55,.35)

    if PDF is not None:
        pdf.savefig(fig)
    ax = None
    fig = None
    return



plot_candidate(mean_intensity, 'mean intensity', labels=label, PDF=pdf)
plot_candidate(percentile_intensity, '75th percentile intensity', labels=label, PDF=pdf)
plot_candidate(frac_nan, 'fraction nan', labels=label, PDF=pdf)
plot_candidate(R_mean, 'R mean', labels=label, PDF=pdf)
plot_candidate(R_mean_fano_factor, 'R mean fano factor', labels=label, PDF=pdf)
plot_candidate(G_mean_fano_factor, 'G mean fano factor', labels=label, PDF=pdf)
plot_candidate(G_mean, 'G mean ', labels=label, PDF=pdf)
plot_candidate(mean_vel, 'Mean Velocity', labels=label, PDF=pdf)
plot_candidate(vel_std, 'std(vel)', labels=label, PDF=pdf)
plot_candidate(std_test_train_ratio, 'std(vel_test) / std(vel_train)', labels=label, PDF=pdf)
plot_candidate(std_vel_test, 'std(vel_test)', labels=label, PDF=pdf)
plot_candidate(vel_pdf_mse, 'MSE of Train vs Test Velocity PDF', labels=label, PDF=pdf)
#plot_candidate(vel_pdf_kl, 'KL divergence of Train vs Test Velocity PDF', labels=label, PDF=pdf)

pdf.close()
print("Finished: " + filename)
