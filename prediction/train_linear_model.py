from models import linear
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter1d

for gtype in ['gcamp', 'gfp']:
    with open('%s_recordings.dat' % gtype, 'rb') as f:
        data = pickle.load(f)
        
    results = {}

    for key in data.keys():
        print("Running "+key)
        time = data[key]['time']
        neurons = data[key]['neurons']
        nderiv = data[key]['neuron_derivatives']
        velocity = data[key]['velocity']
        curvature = data[key]['curvature']

        neurons_and_derivs = np.vstack((neurons, nderiv))

        results[key] = {}
        results[key]['velocity'] = {}
        results[key]['curvature'] = {}
        for bsn in [True, False]:
            results[key]['velocity'][bsn] = linear.optimize(time, neurons_and_derivs, velocity, options_override = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
            results[key]['curvature'][bsn] = linear.optimize(time, neurons_and_derivs, curvature, options_override = {"l1_ratios": [0], "parallelize": False, "best_neuron": bsn})
            print("\tVelocity %s  R^2_ms = %0.2f" % ('(BSN):       ' if bsn else '(Population):', results[key]['velocity'][bsn]['R2ms_test']))
            print("\tCurvature %s R^2_ms = %0.2f" % ('(BSN):       ' if bsn else '(Population):', results[key]['curvature'][bsn]['R2ms_test']))
        
    with open('%s_linear_models.dat' % gtype, 'wb') as f:
        pickle.dump(results, f)