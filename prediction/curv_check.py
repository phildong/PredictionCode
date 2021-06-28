import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle

with open('new_comparison.dat', 'rb') as handle:
    data = pickle.load(handle)#, encoding = 'bytes')

with open('new_comparison_save.dat', 'rb') as f:
    old_data = pickle.load(f)#, encoding = 'bytes')

k = filter(lambda x: '105620' in x, data.keys())[0]

fig, ax = plt.subplots(1, 1, figsize = (25, 10))

ax.set_xlabel('Time', fontsize=20)

ax.set_ylabel(r'Curvature', fontsize = 20)

curv_old = old_data[k]['curvature'][False]['signal']
curv_new = data[k]['curvature'][False]['signal']
time = data[k]['curvature'][False]['time']

ax.plot(time, curv_old-curv_new, label = 'Residual')
# ax.plot(time, curv_new, label = 'New')

fig.legend()

plt.show()
