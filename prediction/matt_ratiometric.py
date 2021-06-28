import scipy.io
from scipy.ndimage import gaussian_filter
import numpy as np
import SLM
import pickle

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

def derivative(neurons):
    nan_zero = np.copy(neurons)
    nan_zero[np.isnan(neurons)] = 0
    nan_zero_filtered = gaussian_filter(nan_zero, order = 1, sigma = (0, 7))

    flat = 0*neurons.copy()+1
    flat[np.isnan(neurons)] = 0
    flat_filtered = gaussian_filter(flat, order = 0, sigma = (0, 7))

    deriv = nan_zero_filtered/flat_filtered

    return deriv

fnames = map(lambda x: 'aml'+str(x)+'_ratiometric.mat', [18, 32])

scores = dict()
for amltype in [32, 18]:
    fname = 'aml'+str(amltype)+'_ratiometric.mat'
    behavior = 'neuron_data'+('_aml18' if amltype == 18 else '')+'.dat'

    print(fname)

    data = scipy.io.loadmat(fname)
    keys = filter(lambda x: x[:5] == 'Brain', data.keys())

    with open(behavior, 'rb') as f:
        behavior_data = pickle.load(f)

    for k in keys[::-1]:
        print("\t"+k)

        worm = data[k][0,0]
        time = behavior_data[k[:-3]]['time']
        vel = behavior_data[k[:-3]]['velocity']
        for i in range(len(worm)):
            print("\t\t"+str(i))
            neurons = worm[i].T

            if k[:-3] in excludeInterval.keys():
                for interval in excludeInterval[k[:-3]]:
                    idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                    idxs_safe = filter(lambda x: x < neurons.shape[1], idxs)
                    neurons = neurons[:,idxs_safe]

            ### unclear why this is necessary
            if neurons.shape[1] > time.shape[0]:
                neurons = neurons[:,:time.shape[0]]

            if neurons.shape[1] < time.shape[0]:
                time = time[:neurons.shape[1]]
                vel = vel[:neurons.shape[1]]
            ###
            
            nderiv = derivative(neurons)
            neurons_and_derivs = np.vstack((neurons, nderiv))

            slm_with_derivs = SLM.optimize_slm(time, neurons_and_derivs, vel, {'l1_ratios': [0]})

            scores[(k, i)] = slm_with_derivs['scorespredicted'][2]
            print("\t\t"+str(scores[(k,i)]))

with open('matt_ratiometric.pkl', 'wb') as f:
    pickle.dump(scores, f)


