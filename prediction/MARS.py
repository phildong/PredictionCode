import numpy as np
from pyearth import Earth

from SLM import split_test

def scores(y, yhat):

    truemean = np.mean(y)
    beta = np.mean(yhat) - truemean
    alpha = np.mean((yhat-truemean)*(y-yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    R2 = 1-np.sum(np.power(y-yhat, 2))/np.sum(np.power(y-truemean, 2))
    rho2 = np.corrcoef(y, yhat)[0,1]**2

    return [R2, rho2 - (alpha+beta**2)**2/((truesigma*predsigma)**2), rho2 - (alpha)**2/((truesigma*predsigma)**2), rho2]

def optimize_mars(time, Xfull, Yfull, options_override = None):
    options = {
    'max_terms': 5,       # Maximum number of MARS terms
    'test_fraction': 0.4  # Fraction of data to use as test set
    }
    if options_override is not None:
        options.update(options_override)
    
    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    model = Earth(max_terms = options['max_terms'])
    model.fit(X.T, Y)

    return {
            'signal': Yfull, 
            'output': model.predict(Xfull.T), 
            'scorespredicted': scores(Y, model.predict(X.T)),
            'time': time, 
            'train_idx': train_idx, 
            'test_idx': test_idx
            }