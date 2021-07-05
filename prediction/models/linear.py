import numpy as np
from sklearn import tree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

from copy import deepcopy
from pathos.multiprocessing import ProcessingPool as Pool

sqdiff = lambda a, b: np.sum(np.power(a-b, 2))

def R2(obs, pred):
    """Returns the coefficient of determination between the observation and prediction"""
    return 1-sqdiff(obs, pred)/sqdiff(obs, np.mean(obs))

def rho2(obs, pred):
    """Returns the Pearson's correlation coefficient between the observation and prediction"""
    return np.corrcoef(obs, pred)[0,1]**2

def R2ms(obs, pred):
    """Returns the mean-subtracted coefficient of determination between the observation and prediction"""
    return R2(obs - np.mean(obs), pred - np.mean(pred))

def train_tree(X, Y, max_depth = 3, min_samples_split = 0.1):
    """Returns a DecisionTreeClassifier predicting the sign of Y using the features X"""

    clf = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split)
    clf = clf.fit(X, np.sign(Y))
    return clf
    
def split_test(X, test_fraction):
    """Returns (train, test), where test is the middle test_fraction of X and train is the rest"""

    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test_fraction/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test_fraction/2

    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T

    return (X_train, X_test)

def optimize(time, Xfullunn, Yfullunn, options_override = None):
    """Find optimal weights for a linear model of type specified by options"""

    options = {
    'derivative_penalty':  False,                       # whether to include a penalty for squared difference of behavior derivative and prediction derivative
    'decision_tree':       False,                       # whether to build a classification tree for the sign of behavior and train positive/negative linear models
    'best_neuron':         False,                       # whether to restrict the model to using only a single neuron or neuron derivative
    'normalize':           True,                        # whether to Z-score neural signals and behaviors before training
    'parallelize':         True,                        # whether to multi-thread the cross-validation stage
    'max_depth':           1,                           # maximum depth of classification tree (ignored if decision_tree = False)
    'lambdas':             np.logspace(-2.5, 0.5, 7),   # list of regularization strengths to test in cross-validation
    'l1_ratios':           [.001, .003, .01, 0.03, .1], # list of L1 penalty ratios to test in cross-validation
    'sigma':               14,                          # width of Gaussian kernel for computing feature time derivatives
    'derivative_weight':   10,                          # weight of derivative penalty (see derivative_penalty option)
    'cv_fraction':         0.4,                         # fraction of training set to use for cross-validation
    'test_fraction':       0.4                          # fraction of data set to use for testing
    }
    options.update(options_override)    

    if options['normalize']: # Z-score features and behavior
        Xmean = np.mean(Xfullunn, axis = 1)[:, np.newaxis]
        Xstd = np.std(Xfullunn, axis = 1)[:, np.newaxis]
        Xfull = (Xfullunn-Xmean)/Xstd
        Yfull = (Yfullunn-np.mean(Yfullunn))/np.std(Yfullunn)
    else:
        Xfull = Xfullunn
        Yfull = Yfullunn

    # Split into training and testing sets
    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    # Define a linear function of features X and parameters P
    if options['best_neuron']:
        f = lambda X, P: P[1]*X + P[0]
    else:
        f = lambda X, P: np.dot(X.T, P[1:]) + P[0]
    
    # Define the ElasticNet regularization penalty (equivalent to an L2 penalty when l1_ratio = 0)
    reg = lambda P, l1_ratio: l1_ratio*np.power(np.sum(np.abs(P[1:])), 2) + 0.5*(1-l1_ratio)*np.sum(P[1:]*P[1:])

    # Define the cost function for training: sum of squared error, regularization, and optional derivative penalty
    error = lambda P, X, Y, alpha, l1_ratio, X_deriv = None, Y_deriv = None: \
        (1./(4*Y.size*np.var(Y)))                                  *sqdiff(Y, f(X, P)) \
      + alpha                                                      *reg(P, l1_ratio) \
      + (options['derivative_weight']*(1/(4*Y.size*np.var(Y_deriv)))*sqdiff(Y_deriv, f(X_deriv, P)) if options['derivative_penalty'] and X_deriv and Y_deriv else 0)

    # Split non-test data into training set and cross-validation set
    X_train, X_cv = split_test(X, options['cv_fraction'])
    Y_train, Y_cv = split_test(Y, options['cv_fraction'])

    # Calculate time derivatives of full training set and cross-validation training set
    X_deriv = gaussian_filter(X, sigma = (0, options['sigma']), order = 1)
    Y_deriv = gaussian_filter(Y, sigma = options['sigma'], order = 1)
    X_train_deriv = gaussian_filter(X_train, sigma = (0, options['sigma']), order = 1)
    Y_train_deriv = gaussian_filter(Y_train, sigma = options['sigma'], order = 1)

    # Define cross-validation functions, with signatures that pathos.multiprocessing can work with
    def crossval(alpha, l1_ratio):
        result = minimize(error, np.zeros(X.shape[0]+1), args=(X_train, Y_train, alpha, l1_ratio, X_train_deriv, Y_train_deriv))
        r2 = R2(Y_cv, f(X_cv, result.x))
        return (r2, alpha, l1_ratio)

    # Define hyperparameter space for cross-validation
    cv_params = [(alpha, l1_ratio) for alpha in options['lambdas'] for l1_ratio in options['l1_ratios']]

    def crossval_unpack(args):
        return crossval(*args)
    
    if not options['decision_tree']:
        if not options['best_neuron']:

            # Test all hyperparameter sets on cross-validation set
            if options['parallelize']:
                p = Pool(processes = len(cv_params))
                r2s = p.map(crossval_unpack, cv_params)
            else:
                r2s = list(map(crossval_unpack, cv_params))

            # Find best-performing hyperparameters
            r2s.sort(key = lambda x: -x[0])
            best_alpha, best_l1_ratio = r2s[0][1:]

            # Use best-performing hyperparameters to train on full training set
            result = minimize(error, np.zeros(X.shape[0]+1), args=(X, Y, best_alpha, best_l1_ratio, X_deriv, Y_deriv))
            P = result.x

            # Compute predicted behavior
            prediction_train = f(X, P)
            prediction_test = f(Xtest, P)
            prediction_full = f(Xfull, P)
            
            return {'weights'        : P[1:],
                    'intercept'      : P[0],
                    'params'         : P,
                    'error'          : result.fun,
                    'alpha'          : best_alpha,
                    'l1_ratio'       : best_l1_ratio,
                    'R2ms_train'     : R2ms(Y, prediction_train),
                    'R2ms_test'      : R2ms(Ytest, prediction_test),
                    'R2_train'       : R2(Y, prediction_train),
                    'R2_test'        : R2(Ytest, prediction_test),
                    'rho2_train'     : rho2(Y, prediction_train),
                    'rho2_test'      : rho2(Ytest, prediction_test),
                    'signal'         : Yfull,
                    'signal_mean'    : np.mean(Yfull),
                    'signal_std'     : np.std(Yfull),
                    'output'         : prediction_full,
                    'time'           : time,
                    'train_idx'      : train_idx,
                    'test_idx'       : test_idx,
                    'crossval'       : r2s
                    }

        else:
            # Try each neuron
            results = [minimize(error, np.zeros(2), args = (X[i,:], Y, 0, 0, X_deriv[i,:], Y_deriv)) for i in np.arange(X.shape[0])]
            r2s = [(R2(Y, f(X[i,:], result.x)), i, result.x, result.fun) for i, result in enumerate(results)]

            # Find best neuron
            r2s.sort(key = lambda x: -x[0])
            r2, idx, params, error = r2s[0]
            
            weights = np.zeros(X.shape[0])
            weights[idx] = params[1]

            P = np.zeros(X.shape[0]+1)
            P[0] = params[0]
            P[1:] = weights

            # Compute predicted behavior
            prediction_train = f(X[idx,:], params)
            prediction_test = f(Xtest[idx,:], params)
            prediction_full = f(Xfull[idx,:], params)

            return {'weights'        : weights,
                    'intercept'      : P[0],
                    'params'         : P,
                    'error'          : error,
                    'alpha'          : 0,
                    'l1_ratio'       : 0,
                    'R2ms_train'     : R2ms(Y, prediction_train),
                    'R2ms_test'      : R2ms(Ytest, prediction_test),
                    'R2_train'       : R2(Y, prediction_train),
                    'R2_test'        : R2(Ytest, prediction_test),
                    'rho2_train'     : rho2(Y, prediction_train),
                    'rho2_test'      : rho2(Ytest, prediction_test),
                    'signal'         : Yfull,
                    'signal_mean'    : np.mean(Yfull),
                    'signal_std'     : np.std(Yfull),
                    'output'         : prediction_full,
                    'time'           : time,
                    'train_idx'      : train_idx,
                    'test_idx'       : test_idx,
                    'crossval'       : None
                    }
    
    else:
        # Build decision tree for sign of behavior
        clf = train_tree(X.T, Y)
        decision = clf.predict(X.T)

        # Split data into positive and negative parts
        X1 = X[:,decision > 0]
        X2 = X[:,decision < 0]
        Y1 = Y[decision > 0]
        Y2 = Y[decision < 0]

        # Build linear models for each part
        new_options = deepcopy(options)
        new_options['decision_tree'] = False
        new_options['test_fraction'] = 0
        res1 = optimize_lm(time, X1, Y1, new_options)
        res2 = optimize_lm(time, X2, Y2, new_options)

        # Assemble predictions together
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

        r2_train = R2(Y, prediction_train)
        r2_test = R2(Ytest, prediction_test)
        
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
                'R2ms_train'     : R2ms(Y, prediction_train),
                'R2ms_test'      : R2ms(Ytest, prediction_test),
                'R2_train'       : R2(Y, prediction_train),
                'R2_test'        : R2(Ytest, prediction_test),
                'rho2_train'     : rho2(Y, prediction_train),
                'rho2_test'      : rho2(Ytest, prediction_test),
                'signal'         : Yfull,
                'signal_mean'    : np.mean(Yfull),
                'signal_std'     : np.std(Yfull),
                'output'         : prediction_full,
                'res_pos'        : res1,
                'res_neg'        : res2,
                'time'           : time,
                'train_idx'      : train_idx,
                'test_idx'       : test_idx
                }