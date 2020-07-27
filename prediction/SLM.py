import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from copy import deepcopy

from Classifier import rectified_derivative

import dataHandler as dh
import userTracker

from pathos.multiprocessing import ProcessingPool as Pool

def shift_sqdiff(a, b, width = 6):
    sqdiffs = np.zeros((2*width+1, a.size))
    for j in np.arange(-width, width+1):
        sqdiffs[j+width,:] = np.power(np.roll(a, j)-b, 2)
    
    return np.sum(np.min(sqdiffs, axis=0))

def train_tree(X, Y):
    clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=.1)
    clf = clf.fit(X, np.sign(Y))
    return clf

def R2(P, f, X, Y, width = 6):
    return 1-shift_sqdiff(Y, f(X, P))/shift_sqdiff(Y, np.mean(Y), width=width)

def rho(P, f, X, Y):
    return np.corrcoef(Y, f(X, P))

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def optimize_slm(time, Xfullunn, Yfull, options = None):
    if options is None:
        options = dict()
    default_options = {
    'derivative_penalty': False,
    'decision_tree': False,
    'best_neuron': False,
    'normalize': True,
    'parallelize': True,
    'max_depth': 1,
    'alphas': np.logspace(-1.5, 2.5, 7),
    'l1_ratios': np.linspace(0, 1, 5),
    'sigma': 14,
    'derivative_weight': 10,
    'cv_fraction': 0.4,
    'test_fraction': 0.4,
    'time_shift': 0
    }
    for k in default_options:
        if k not in options:
            options[k] = default_options[k]

    if options['normalize']:
        Xmean = np.mean(Xfullunn, axis = 1)[:, np.newaxis]
        Xstd = np.std(Xfullunn, axis = 1)[:, np.newaxis]
        Xfull = (Xfullunn-Xmean)/Xstd
    else:
        Xfull = Xfullunn

    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    if options['best_neuron']:
        f = lambda X, P: P[1]*X + P[0]
    else:
        f = lambda X, P: np.dot(X.T, P[1:]) + P[0]
    reg = lambda P, l1_ratio: l1_ratio*np.sum(np.abs(P[1:])) + 0.5*(1-l1_ratio)*np.sqrt(np.sum(P[1:]*P[1:]))

    error_plain = lambda P, X, Y, alpha, l1_ratio: (1./(4*Y.size*np.var(Y)))*shift_sqdiff(Y, f(X, P), width=options['time_shift']) + alpha*reg(P, l1_ratio)
    if options['derivative_penalty']:
        error = lambda P, X, Y, X_deriv, Y_deriv, alpha, l1_ratio: error_plain(P, X, Y, alpha, l1_ratio) + options['derivative_weight']*(1/(4*Y.size*np.var(Y_deriv)))*shift_sqdiff(Y_deriv, f(X_deriv, P), width=options['time_shift'])
    else:
        error = error_plain

    X_train, X_cv = split_test(X, options['cv_fraction'])
    Y_train, Y_cv = split_test(Y, options['cv_fraction'])
    
    if not options['decision_tree']:
        if not options['best_neuron']:
            if not options['derivative_penalty']:
                def crossval(alpha, l1_ratio):
                    result = minimize(error, np.zeros(X.shape[0]+1), args=(X_train, Y_train, alpha, l1_ratio))
                    r2 = R2(result.x, f, X_cv, Y_cv, width=options['time_shift']) 
                    return (r2, alpha, l1_ratio)

                def crossval_unpack(args):
                    return crossval(*args)

                params = [(alpha, l1_ratio) for alpha in options['alphas'] for l1_ratio in options['l1_ratios']]

                if options['parallelize']:
                    p = Pool(processes = len(options['alphas'])*len(options['l1_ratios']))
                    r2s = p.map(crossval_unpack, params)
                else:
                    r2s = list(map(crossval_unpack, params))

                r2s.sort(key = lambda x: -x[0])

                best_alpha = r2s[0][1]
                best_l1_ratio = r2s[0][2]
                result = minimize(error, np.zeros(X.shape[0]+1), args=(X, Y, best_alpha, best_l1_ratio))
                P = result.x
                
                return {'weights'        : P[1:],
                        'intercepts'     : P[0],
                        'params'         : P,
                        'error'          : result.fun,
                        'alpha'          : best_alpha,
                        'l1_ratio'       : best_l1_ratio,
                        'score'          : R2(P, f, X, Y, width=options['time_shift']),
                        'scorepredicted' : R2(P, f, Xtest, Ytest, width=options['time_shift']),
                        'corr'           : rho(P, f, X, Y)[0,1],
                        'corrpredicted'  : rho(P, f, Xtest, Ytest)[0,1],
                        'signal'         : Yfull,
                        'output'         : f(Xfull, P),
                        'time'           : time,
                        'train_idx'      : train_idx,
                        'test_idx'       : test_idx,
                        'variance'       : np.var(X, axis=1),
                        'crossval'       : r2s
                        }
            
            else:
                X_deriv = gaussian_filter(X, sigma = (0, options['sigma']), order=1)
                Y_deriv = gaussian_filter(Y, sigma = options['sigma'], order = 1)
                X_train_deriv = gaussian_filter(X_train, sigma = (0, options['sigma']), order = 1)
                Y_train_deriv = gaussian_filter(Y_train, sigma = options['sigma'], order = 1)

                def crossval(alpha, l1_ratio):
                    result = minimize(error, np.zeros(X.shape[0]+1), args=(X_train, Y_train, X_train_deriv, Y_train_deriv, alpha, l1_ratio))
                    r2 = R2(result.x, f, X_cv, Y_cv, width=options['time_shift']) 
                    return (r2, alpha, l1_ratio)

                def crossval_unpack(args):
                    return crossval(*args)
                
                params = [(alpha, l1_ratio) for alpha in options['alphas'] for l1_ratio in options['l1_ratios']]

                if options['parallelize']:
                    p = Pool(processes = len(options['alphas'])*len(options['l1_ratios']))
                    r2s = p.map(crossval_unpack, params)
                else:
                    r2s = list(map(crossval_unpack, params))

                r2s.sort(key = lambda x: -x[0])

                best_alpha = r2s[0][1]
                best_l1_ratio = r2s[0][2]
                result = minimize(error, np.zeros(X.shape[0]+1), args=(X, Y, X_deriv, Y_deriv, best_alpha, best_l1_ratio))
                P = result.x

                return {'weights'        : P[1:],
                        'intercepts'     : P[0],
                        'params'         : P,
                        'error'          : result.fun,
                        'alpha'          : best_alpha,
                        'l1_ratio'       : best_l1_ratio,
                        'score'          : R2(P, f, X, Y, width=options['time_shift']),
                        'scorepredicted' : R2(P, f, Xtest, Ytest, width=options['time_shift']),
                        'corr'           : rho(P, f, X, Y)[0,1],
                        'corrpredicted'  : rho(P, f, Xtest, Ytest)[0,1],
                        'signal'         : Yfull,
                        'output'         : f(Xfull, P),
                        'time'           : time,
                        'train_idx'      : train_idx,
                        'test_idx'       : test_idx,
                        'variance'       : np.var(X, axis=1),
                        'crossval'       : r2s
                        }

        else:
            if not options['derivative_penalty']:
                best_neuron_idx = -1
                best_neuron_R2 = -np.inf
                best_neuron_params = np.zeros(2)
                best_neuron_error = np.inf

                for i in np.arange(X.shape[0]):
                    neuron = X[i,:]
                    result = minimize(error, np.zeros(2), args = (neuron, Y, 0, 0))
                    r2 = R2(result.x, f, neuron, Y, width=options['time_shift'])
                    if r2 > best_neuron_R2:
                        best_neuron_idx = i
                        best_neuron_params = result.x
                        best_neuron_R2 = r2
                        best_neuron_error = result.fun
                
                weights = np.zeros(X.shape[0])
                weights[best_neuron_idx] = best_neuron_params[1]
                P = np.zeros(X.shape[0]+1)
                P[0] = best_neuron_params[0]
                P[1:] = weights

                return {'weights'        : weights,
                        'intercepts'     : best_neuron_params[0],
                        'params'         : P,
                        'error'          : best_neuron_error,
                        'alpha'          : 0,
                        'l1_ratio'       : 0,
                        'score'          : best_neuron_R2,
                        'scorepredicted' : R2(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest, width=options['time_shift']),
                        'corr'           : rho(best_neuron_params, f, X[best_neuron_idx,:], Y)[0,1],
                        'corrpredicted'  : rho(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest)[0,1],
                        'signal'         : Yfull,
                        'output'         : f(Xfull[best_neuron_idx,:], best_neuron_params),
                        'time'           : time,
                        'train_idx'      : train_idx,
                        'test_idx'       : test_idx,
                        'variance'       : np.var(X, axis=1)
                        }
            
            else:
                X_deriv = gaussian_filter(X, sigma = (0, options['sigma']), order=1)
                Y_deriv = gaussian_filter(Y, sigma = options['sigma'], order = 1)

                best_neuron_idx = -1
                best_neuron_R2 = -np.inf
                best_neuron_params = np.zeros(2)
                best_neuron_error = np.inf

                for i in np.arange(X.shape[0]):
                    neuron = X[i,:]
                    neuron_deriv = X_deriv[i,:]
                    result = minimize(error, np.zeros(2), args = (neuron, Y, neuron_deriv, Y_deriv, 0, 0))
                    r2 = R2(result.x, f, neuron, Y, width=options['time_shift'])
                    if r2 > best_neuron_R2:
                        best_neuron_idx = i
                        best_neuron_params = result.x
                        best_neuron_R2 = r2
                        best_neuron_error = result.fun
                
                weights = np.zeros(X.shape[0])
                weights[best_neuron_idx] = best_neuron_params[1]
                P = np.zeros(X.shape[0]+1)
                P[0] = best_neuron_params[0]
                P[1:] = weights

                return {'weights'        : weights,
                        'intercepts'     : best_neuron_params[0],
                        'params'         : P,
                        'error'          : best_neuron_error,
                        'alpha'          : 0,
                        'l1_ratio'       : 0,
                        'score'          : best_neuron_R2,
                        'scorepredicted' : R2(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest, width=options['time_shift']),
                        'corr'           : rho(best_neuron_params, f, X[best_neuron_idx,:], Y)[0,1],
                        'corrpredicted'  : rho(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest)[0,1],
                        'signal'         : Yfull,
                        'output'         : f(Xfull[best_neuron_idx,:], best_neuron_params),
                        'time'           : time,
                        'train_idx'      : train_idx,
                        'test_idx'       : test_idx,
                        'variance'       : np.var(X, axis=1)
                        }
    
    else:
        clf = train_tree(X.T, Y)
        decision = clf.predict(X.T)

        X1 = X[:,decision > 0]
        X2 = X[:,decision < 0]
        Y1 = Y[decision > 0]
        Y2 = Y[decision < 0]

        new_options = deepcopy(options)
        new_options['decision_tree'] = False
        new_options['test_fraction'] = 0
        res1 = optimize_slm(time, X1, Y1, new_options)
        res2 = optimize_slm(time, X2, Y2, new_options)

        prediction_train = np.zeros(Y.size)
        prediction_train[decision > 0] = np.dot(X1.T, res1['weights']) + res1['intercepts']
        prediction_train[decision < 0] = np.dot(X2.T, res2['weights']) + res2['intercepts']

        decision_test = clf.predict(Xtest.T)
        prediction_test = np.zeros(Ytest.size)
        prediction_test[decision_test > 0] = np.dot(Xtest[:,decision_test>0].T, res1['weights']) + res1['intercepts']
        prediction_test[decision_test < 0] = np.dot(Xtest[:,decision_test<0].T, res2['weights']) + res2['intercepts']

        decision_full = clf.predict(Xfull.T)
        prediction_full = np.zeros(Yfull.size)
        prediction_full[decision_full > 0] = np.dot(Xfull[:,decision_full>0].T, res1['weights']) + res1['intercepts']
        prediction_full[decision_full < 0] = np.dot(Xfull[:,decision_full<0].T, res2['weights']) + res2['intercepts']

        r2_train = 1-shift_sqdiff(Y, prediction_train, width=options['time_shift'])/shift_sqdiff(Y, np.mean(Y), width=options['time_shift'])
        r2_test = 1-shift_sqdiff(Ytest, prediction_test, width=options['time_shift'])/shift_sqdiff(Y, np.mean(Y), width=options['time_shift'])
        
        corr_train = np.corrcoef(Y, prediction_train)[0,1]
        corr_test = np.corrcoef(Ytest, prediction_test)[0,1]

        return {'decision'       : decision_full,
                'weights_pos'    : res1['weights'],
                'weights_neg'    : res2['weights'],
                'intercepts_pos' : res1['intercepts'],
                'intercepts_neg' : res2['intercepts'],
                'error'          : (Y1.size*res1['error']+Y2.size*res2['error'])/Y.size,
                'alpha_pos'      : res1['alpha'],
                'alpha_neg'      : res2['alpha'],
                'l1_ratio_pos'   : res1['l1_ratio'],
                'l1_ratio_neg'   : res2['l1_ratio'],
                'score'          : r2_train,
                'scorepredicted' : r2_test,
                'corr'           : corr_train,
                'corrpredicted'  : corr_test,
                'signal'         : Yfull,
                'output'         : prediction_full,
                'res_pos'        : res1,
                'res_neg'        : res2,
                'time'           : time,
                'train_idx'      : train_idx,
                'test_idx'       : test_idx,
                'variance'       : np.var(X, axis=1)
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
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            nderiv_pos, nderiv_neg, _ = rectified_derivative(neurons)
            X = np.vstack((neurons, nderiv_pos, nderiv_neg))

            settings = [(j, k) for j in range(2) for k in range(2)]
            r2s = {}
            for s in settings:
                res = optimize_slm(time, X, velocity, options = {'decision_tree': s[0], 'best_neuron': s[1], 'l1_ratios': [.9]})
                r2s[s] = res
                print(s, res['score'], res['scorepredicted'])
            
            output_data[typ_cond+" "+key] = r2s
            
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))

            r2keys = list(r2s.keys())
            labels = list(map(str,r2keys))
            r2s_train = [r2s[k]['score'] for k in r2keys]
            r2s_test  = [r2s[k]['scorepredicted'] for k in r2keys]

            x = np.arange(len(r2s))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, r2s_train, width, label=r'$R^2_{\mathrm{train}}$')
            rects2 = ax.bar(x + width/2, r2s_test, width, label=r'$R^2_{\mathrm{test}}$')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('$R^2$')
            ax.set_title(typ_cond+" "+key)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()


            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{:.1f}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3*np.sign(height)),  # 3 points vertical offset
                                fontsize=10,
                                textcoords="offset points",
                                ha='center', va='bottom')


            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()
            pdf.savefig(fig)
    
    pdf.close()

    import pickle
    with open('aks_regression_results.dat', 'wb') as handle:
        pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)