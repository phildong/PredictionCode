import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree, ensemble
from scipy.optimize import minimize
from scipy.stats import mode
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import os
from copy import deepcopy

import dataHandler as dh
import userTracker

def rectified_derivative(neurons):
    deriv = gaussian_filter(neurons, order = 1, sigma = (0, 14))
    deriv[deriv < 0] = 0
    return deriv

def score(predict, true):
    return 1-(np.count_nonzero(predict-true)+.0)/np.count_nonzero(mode(true).mode-true)

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def optimize_clf(time, Xfull, Yfull, options = None):
    if options is None:
        options = dict()
    default_options = {
    'test_fraction': 0.4,
    'C'            : 0.001
    }
    for k in default_options:
        if k not in options:
            options[k] = default_options[k]

    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    clf = tree.DecisionTreeClassifier(max_depth=3,min_samples_split=.1)
    clf.fit(X.T, np.sign(Y))

    train_output = clf.predict(X.T)
    test_output = clf.predict(Xtest.T)

    return {'score'          : score(train_output, np.sign(Y)),
            'scorepredicted' : score(test_output, np.sign(Ytest)),
            'signal'         : Yfull,
            'output'         : clf.predict(Xfull.T),
            'time'           : time,
            'train_idx'      : train_idx,
            'test_idx'       : test_idx
            }


if __name__ == '__main__':
    output_data = {}

    for typ_cond in ['AKS297.51_moving', 'AML32_moving', 'AML70_chip', 'AML70_moving', 'AML18_moving']:
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
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            nderiv = rectified_derivative(neurons)

            if key[-6:] == '110803':
                # plt.plot(neurons[13,:])
                fig, ax = plt.subplots(1, 1, figsize=(10,10))
                ax.plot(velocity, nderiv[13,:], 'k.')
                ax.set_xlabel('CMS Velocity')
                ax.set_ylabel('Rectified Derivative #13')
                ax.set_title(key+' Neuron #13')

                fig.tight_layout()
                fig.savefig('tuning_13_110803.png')

            velocity_res = optimize_clf(time, nderiv, velocity)
            print(key, 'velocity', velocity_res['score'], velocity_res['scorepredicted'])

            curvature_res = optimize_clf(time, nderiv, curvature)
            print(key, 'curvature', curvature_res['score'], curvature_res['scorepredicted'])
            
            output_data[typ_cond+" "+key] = {'velocity': velocity_res, 'curvature': curvature_res}

    import pickle
    with open('clf_results.dat', 'wb') as handle:
        pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)