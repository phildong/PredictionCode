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

with open('comparison_results.dat', 'rb') as handle:
    data = pickle.load(handle)

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

neuron_data = {}
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

        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
        
        neuron_data[key] = neurons

keys = list(data.keys())
keys.sort()

figtypes = ['bsn', 'slm']

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "slm_weight_results.pdf"))

for key in keys:

    fig = plt.figure(constrained_layout = True, figsize=(10*(len(figtypes)+2), 10*len(figtypes)))
    gs = gridspec.GridSpec(len(figtypes), len(figtypes)+2, figure=fig, width_ratios=[1]*(len(figtypes)+2))

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

        sc = fig.add_subplot(gs[row, 1])
        sc.plot(res['signal'][res['train_idx']], res['output'][res['train_idx']], 'go', label = 'Train', rasterized = True)
        sc.plot(res['signal'][res['test_idx']], res['output'][res['test_idx']], 'bo', label = 'Test', rasterized = True)
        sc.plot([min(res['signal']), max(res['signal'])], [min(res['signal']), max(res['signal'])], 'k-.')
        sc.set_title(figtype+r' $\rho^2_{\mathrm{adj},2}(\mathrm{velocity})$ = %0.3f' % (res['corrpredicted']**2 - (alpha)**2/((truesigma*predsigma)**2)))
        sc.set_xlabel('Measured Velocity')
        sc.set_ylabel('Predicted Velocity')
        sc.legend()

    ax = fig.add_subplot(gs[:, 2:])

    slm_weights = data[key]['slm']['weights']*np.sqrt(data[key]['slm']['variance'])
    slm_weights_raw = data[key]['slm']['weights']
    correlations = [np.corrcoef(x, data[key]['slm']['signal'])[0,1] for x in neuron_data[key]]

    ax.scatter(correlations, slm_weights, label="scaled by variance")
    ax.scatter(correlations, slm_weights_raw*100, label="Raw", marker="x", color="red")
    ax.axvline(0, linestyle='dashed')
    ax.axhline(0, linestyle='dashed')
    ax.set_xlabel(r'$\rho$', fontsize=14)
    ax.set_ylabel('Weight', fontsize=14)

    fig.suptitle(key)
    fig.tight_layout(rect=[0,.03,1,0.97])

    pdf.savefig(fig)

pdf.close()