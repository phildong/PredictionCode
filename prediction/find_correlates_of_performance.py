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
        label.append(key)




def plot_candidate(x,  x_name, metric  = 'rho2', metric_name = 'rho2', labels=label):
    fig, ax = plt.subplots()
    ax.scatter(x, rho2)
    ax.set_xlabel(x_name)
    ax.set_ylabel(metric_name)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (mean_intensity[i], rho2[i]))



plot_candidate(mean_intensity, 'mean intensity', labels=label)
plot_candidate(percentile_intensity, '75th percentile intensity', labels=label)
plot_candidate(frac_nan, 'fraction nan', labels=label)
plot_candidate(R_mean, 'R mean', labels=label)
plot_candidate(R_mean_fano_factor, 'R mean fano factor', labels=label)
plot_candidate(G_mean_fano_factor, 'G mean fano factor', labels=label)
plot_candidate(G_mean, 'G mean ', labels=label)
plt.show()
