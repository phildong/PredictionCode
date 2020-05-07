import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from pyearth import Earth
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from copy import deepcopy

from Classifier import rectified_derivative
from SLM import split_test

import dataHandler as dh
import userTracker


def score(predict, true):
    return 1-np.sum(np.power(predict-true, 2))/np.sum(np.power(true-np.mean(true), 2))

def optimize_mars(time, Xfull, Yfull, options = None):
    if options is None:
        options = dict()
    default_options = {
    'max_terms': 5,
    'test_fraction': 0.4
    }
    for k in default_options:
        if k not in options:
            options[k] = default_options[k]
    
    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    model = Earth(max_terms = options['max_terms'])
    model.fit(X.T, Y)

    return {
            'signal': Yfull, 
            'output': model.predict(Xfull.T), 
            'score': score(model.predict(X.T), Y),
            'scorepredicted': score(model.predict(Xtest.T), Ytest),
            'time': time, 
            'train_idx': train_idx, 
            'test_idx': test_idx
            }

if __name__ == '__main__':
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_test.pdf"))

    output_data = {}

    for typ_cond in ['AKS297.51_moving', 'AML32_moving']:#, 'AML70_chip', 'AML70_moving', 'AML18_moving']:
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
            print(key)
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
            
            output_data[typ_cond+" "+key] = optimize_mars(time, neurons, velocity, options={'max_terms': 5 if key[:3] == 'AML' else 2})

    import pickle
    with open('mars_results.dat', 'wb') as handle:
        pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)