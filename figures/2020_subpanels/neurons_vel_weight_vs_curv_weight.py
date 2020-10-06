import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.backends.backend_pdf
from prediction import userTracker
from prediction import dataHandler as dh
import os
from scipy.ndimage import gaussian_filter
import prediction.provenance as prov



behaviors = ['velocity', 'curvature']
slm_weights_raw = [None] * 2
slm_weights_raw_deriv = [None] * 2

for k, behavior in enumerate(behaviors):

    pickled_data = '/projects/LEIFER/PanNeuronal/decoding_analysis/analysis/comparison_results_' + behavior + '_l10.dat'
    with open(pickled_data, 'rb') as handle:
        data = pickle.load(handle)

    excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
    excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]],
                       'BrainScanner20200309_151024': [[125, 135], [30, 40]],
                       'BrainScanner20200309_153839': [[35, 45], [160, 170]],
                       'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                       'BrainScanner20200130_105254': [[65, 75]],
                       'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

    for typ_cond in ['AKS297.51_moving']:
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
            if behavior == 'velocity':
                beh = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
            elif behavior == "curvature":
                beh = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
            else:
                assert False


            if key in excludeInterval.keys():
                for interval in excludeInterval[key]:
                    idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                    time = time[idxs]
                    neurons = neurons[:,idxs]
                    beh = beh[idxs]




    key='BrainScanner20200130_110803'



    slm_weights_raw[k] = np.abs(data[key]['slm_with_derivs']['weights'][:data[key]['slm_with_derivs']['weights'].size / 2])
    slm_weights_raw_deriv[k] = np.abs(data[key]['slm_with_derivs']['weights'][data[key]['slm_with_derivs']['weights'].size / 2:])

import os
outfilename = key + 'weights_vel_v_curve.pdf'
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), outfilename))


fig1 = plt.figure(constrained_layout=True, figsize=[10, 10])
#find smallest weight and alrgest weight
allweights =  np.concatenate([np.concatenate(slm_weights_raw_deriv), np.concatenate(slm_weights_raw)])
lw = np.max(allweights)


#Generate scatter plot and box plot
f1_ax1 = fig1.add_subplot(111, xlabel='Magnitude of Weight for ' + behaviors[1], ylabel='Magnitude of Weight for '+ behaviors[0])
f1_ax1.plot(slm_weights_raw[1], slm_weights_raw[0], 'o', label='F')
f1_ax1.plot(slm_weights_raw_deriv[1], slm_weights_raw_deriv[0], 'o', label='dF/dt')
f1_ax1.plot([0, lw], [0, lw], 'k--')
fig1.legend()
prov.stamp(f1_ax1, .55, .35, __file__ + '\n' + pickled_data)
pdf.savefig(fig1)

fig2 = plt.figure(constrained_layout=True, figsize=[10, 10])
fig2.suptitle('max(F, dF/dt)')
f2_ax1 = fig2.add_subplot(111, xlabel='Magnitude of Weight for ' + behaviors[1], ylabel='Magnitude of Weight for '+ behaviors[0])
f2_ax1.plot(np.max([[slm_weights_raw[1]], [slm_weights_raw_deriv[1]]], axis=0).T, np.max([[slm_weights_raw[0]], [slm_weights_raw_deriv[0]]], axis=0).T, 'o', label='max(F,dF/dt)')
f2_ax1.plot([0, lw], [0, lw], 'k--')
prov.stamp(f2_ax1, .55, .35, __file__ + '\n' + pickled_data)
pdf.savefig(fig2)



pdf.close()
print("wrote " + outfilename)

