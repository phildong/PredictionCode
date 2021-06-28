import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

with open('new_comparison_aml18.dat', 'rb') as f:
    results_aml18 = pickle.load(f)
with open('new_comparison.dat', 'rb') as f:
    results = pickle.load(f)

res = results['BrainScanner20200130_110803']['velocity'][(False, True, False)]
print(res['scorespredicted'][1])

res18 = results_aml18['BrainScanner20200204_102136']['velocity'][(False, True, False)]
print(res18['scorespredicted'][1])

plt.hist(res18['weights'], bins = 30, alpha = 0.5)
plt.hist(res['weights'], bins = 30, alpha = 0.5)
plt.show()