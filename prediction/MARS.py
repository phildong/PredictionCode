import numpy as np
from pyearth import Earth

from SLM import R2, R2ms, rho, split_test

def optimize_mars(time, Xfull, Yfull, options_override = None):
    options = {
    'max_terms': 5,       # Maximum number of MARS terms
    'test_fraction': 0.4  # Fraction of data to use as test set
    }
    options.update(options_override)
    
    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])
    train_idx, test_idx = split_test(np.arange(Yfull.size), options['test_fraction'])

    model = Earth(max_terms = options['max_terms'])
    model.fit(X.T, Y)

    return {
            'signal': Yfull, 
            'output': model.predict(Xfull.T), 
            'R2ms_train': R2ms(model.predict(X.T), Y),
            'R2ms_test': R2ms(model.predict(Xtest.T), Ytest),
            'R2_train': R2(model.predict(X.T), Y),
            'R2_test': R2(model.predict(Xtest.T), Ytest),
            'rho_train': rho(model.predict(X.T), Y),
            'rho_test': rho(model.predict(Xtest.T), Ytest),
            'time': time, 
            'train_idx': train_idx, 
            'test_idx': test_idx
            }