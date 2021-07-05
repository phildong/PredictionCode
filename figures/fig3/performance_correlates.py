import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.backends.backend_pdf

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import kl_div
from sklearn.preprocessing import MinMaxScaler

import pickle
import os

from utility import user_tracker
from utility import data_handler as dh


def calc_pdf(x, low_lim, high_lim, nbins):
    counts, bin_edges = np.histogram(x, np.linspace(low_lim, high_lim, nbins))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    density = np.true_divide(counts, np.sum(counts))
    return density, bin_centers, bin_edges

def compare_pdf(a, b, low_lim=-3, high_lim=3, nbins=24, alabel="", blabel="", PDF=None, suplabel=""):
    a_hist, a_bin_centers, a_bin_edges = calc_pdf(a, low_lim, high_lim, nbins)
    b_hist, bin_centers, bin_edges = calc_pdf(b, low_lim, high_lim, nbins)
    assert np.all(a_bin_edges==bin_edges)

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

for behavior in ['velocity', 'curvature']:
    filename = 'correlates_of_performance_' + behavior + '.pdf'
    with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as handle:
        data = pickle.load(handle)

    mean_intensity = []
    percentile_intensity = []
    frac_nan = []
    rho2_adj = []
    bsn_rho2_adj = []
    recording_length = []
    label = []
    R_mean = []
    R_mean_fano_factor = []
    G_mean = []
    G_mean_fano_factor = []
    G_max_fano_factor = []
    G_std_G_mean_largest_neuron = []
    G_std_G_mean_97_neuron = []
    G_R_ratio = []
    G_R_percentile_ratio = []
    G_R_percentile_ratio_percentile_neuron = []
    G_R_mean_ratio_percentile_neuron = []
    G_R_mean_ratio_percentile2_neuron = []
    mean_vel = []
    vel_std = []
    std_test_train_ratio = []
    std_vel_test = []
    vel_pdf_mse = []
    vel_pdf_kl =  []

    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

    excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

    for typ_cond in ['AKS297.51_moving', 'AML32_moving']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '%s/' % typ_cond)
        dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

        dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
        keyList = np.sort(dataSets.keys())

        for key in keyList:
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            neurons_raw = dataSets[key]['Neurons']['I_smooth']
            R = dataSets[key]['Neurons']['RedMatlab']
            G = dataSets[key]['Neurons']['GreenMatlab']
            vel = dataSets[key]['BehaviorFull']['CMSVelocity']
            res = data[key][behavior][False]
            bsnres = data[key][behavior][True]
            beh = res['signal'][res['train_idx']]
            beh_test = res['signal'][res['test_idx']]



            if key in excludeInterval.keys():
                for interval in excludeInterval[key]:
                    idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                    time = time[idxs]
                    neurons = neurons[:,idxs]
                    neurons_raw = neurons_raw[:,idxs]
                    R = R[:,idxs]
                    G = G[:,idxs]
                    vel = vel[idxs]

            rho2_adj.append(res['R2ms_test']
            bsn_rho2_adj.append(bsnres['R2ms_test'])

            mean_intensity.append(np.nanmean(neurons))
            percentile_intensity.append(np.nanpercentile(neurons, 75))
            frac_nan.append(np.true_divide(np.sum(np.sum(np.isnan(neurons_raw))), neurons_raw.size))
            recording_length.append(neurons.shape[1])
            R_mean.append(np.nanmean(R))
            R_mean_fano_factor.append(np.nanmedian((np.nanstd(R, 1)**2 / np.nanmean(R, 1) )) )
            G_mean.append(np.nanmean(G))
            G_mean_fano_factor.append(np.nanmedian((np.nanstd(G, 1)**2 / np.nanmean(G, 1) )) )
            G_max_fano_factor.append(np.max(np.nanstd(G, 1)**2/np.nanmean(G,1)) )
            G_std_G_mean_largest_neuron.append(np.max(np.nanstd(G, 1)/np.nanmean(G,1)))
            G_std_G_mean_97_neuron.append(np.nanpercentile(np.nanstd(G, 1) / np.nanmean(G, 1), 97))
            G_R_ratio.append(np.nanmean(np.nanmean(np.true_divide(G, R), 1)))
            G_R_percentile_ratio.append(np.true_divide(np.nanpercentile(G, 90),  np.nanpercentile(R, 90)))
            G_R_percentile_ratio_percentile_neuron.append(np.nanpercentile(np.true_divide(np.nanpercentile(G, 90, axis=1), np.nanpercentile(R, 90, axis=1)), 90))
            G_R_mean_ratio_percentile_neuron.append(np.nanmean(np.true_divide(np.nanpercentile(G, 90, axis=1), np.nanpercentile(R, 90, axis=1))))
            G_R_mean_ratio_percentile2_neuron.append(np.nanmean(np.true_divide(np.nanpercentile(G, 95, axis=1), np.nanpercentile(R, 95, axis=1))))

            mean_vel.append(np.nanmean(vel))
            vel_std.append(np.nanstd(vel))
            std_test_train_ratio.append(np.nanstd(beh_test) / np.nanstd(beh))
            std_vel_test.append(np.nanstd(beh_test))
            mse = compare_pdf(beh_test, beh, alabel="test", blabel="train", suplabel=key + "rho2 = %.2f " % data[key][behavior][type_helper('slm_with_derivs')]['scorespredicted'][2], PDF=pdf)
            vel_pdf_mse.append(mse)
            #vel_pdf_kl.append(kl)

            label.append(key[12:])
            
def plot_candidate(x,  x_name, metric  = 'rho2', metric_name = 'rho2_adj', labels=label, PDF=None, ylim=[0,1]):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(x, rho2_adj, 'o', markersize=16)

    #adapted from: https://stackoverflow.com/questions/18767523/fitting-data-with-numpy
    try:
        coefs = np.polyfit(x, rho2_adj, 1)
        x_new = np.linspace(np.min(x), np.max(x), num=len(x))
        ffit = np.polyval(x_new, coefs)
        plt.plot(x_new, ffit,'r--')
        print("plotted the fit")
    except:
        None

    ax.set_xlabel(x_name)
    ax.set_ylabel(metric_name)
    ax.set_xlim(np.nanmin(x)*.9, np.nanmax(x)*1.4)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title('Corrcoef =%.2f' %np.corrcoef(x, rho2_adj)[0,1] )
    ax.tick_params(axis='both', which='major', labelsize=19)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], rho2_adj[i]))

    if PDF is not None:
        pdf.savefig(fig)
        
    plt.close(fig)



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
plot_candidate(vel_pdf_mse, 'MSE of Train vs Test' + behavior + ' PDF', labels=label, PDF=pdf)
plot_candidate(G_R_ratio, " mean of per neuron mean of G/R", labels=label, PDF=pdf)
plot_candidate(G_R_percentile_ratio, " 90th all G / 90th perecentile all R", labels=label, PDF=pdf)
plot_candidate(G_R_percentile_ratio_percentile_neuron, " 90th percentile neuron of 90th percentile G / 90th perecentile R", labels=label, PDF=pdf)
plot_candidate(G_R_mean_ratio_percentile_neuron, " mean across neurons of 90th percentile G / 90th perecentile R", labels=label, PDF=pdf)
plot_candidate(G_R_mean_ratio_percentile2_neuron, " mean across neurons of 95th percentile G / 95th perecentile R", labels=label, PDF=pdf)
plot_candidate(bsn_rho2_adj, " Best Single Neuron rho2_adj",
                   labels=label, PDF=pdf)
plot_candidate(G_std_G_mean_largest_neuron, " value for Green neuron with highest std/mean", labels=label, PDF=pdf)
plot_candidate(G_std_G_mean_97_neuron, " value for Green neuron with 97th percentile highest std/mean", labels=label, PDF=pdf)
plot_candidate(G_max_fano_factor, " Fano Factor for GCaMP Neuron with highest Fano Factor", labels=label, PDF=pdf)

#plot_candidate(vel_pdf_kl, 'KL divergence of Train vs Test Velocity PDF', labels=label, PDF=pdf)

pdf.close()
print("Finished: " + filename)
