import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf

import os
import numpy as np
import numpy.ma as ma
import numpy.matlib

from skimage.util.shape import view_as_windows as viewW

from utility import user_tracker
from utility import data_handler as dh

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    # This function will roll each row of a matrix a, a an amount specified by r.
    # I got it here: https://stackoverflow.com/a/51613442/200688
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]

def nancorrcoef(A, B):
    a = ma.masked_invalid(A)
    b = ma.masked_invalid(B)

    msk = (~a.mask & ~b.mask)

    return ma.corrcoef(a[msk], b[msk])

with open('%s/gcamp_recordings.dat' % user_tracker.codePath(), 'rb') as f:
    neuron_data = pickle.load(f)

key = 'BrainScanner20200130_110803'

dset = neuron_data[key]

activity = dset['neurons']
deriv = dset['neuron_derivatives']
time = dset['time']
vel = dset['velocity']
curv = dset['curvature']

numNeurons = activity.shape[0]

AVAR = 32
AVAL = 15

#Loop through each neuron
for neuron in (AVAR, AVAL):
    print("Generating plot for neuron %d" % neuron)

    fig = plt.figure(constrained_layout=True, figsize=[6, 8])
    gs = gridspec.GridSpec(ncols=4, nrows=5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1d = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2d = fig.add_subplot(gs[0, 3], sharex=ax2)
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    ax5 = fig.add_subplot(gs[3, :], sharex=ax3)
    ax6 = fig.add_subplot(gs[4, :], sharex=ax3)

    ax = [ax1, ax2, ax3, ax4, ax5, ax6]

    fig.suptitle(key + ' ' + idn + ' Neuron: #' + str(neuron))

    if PLOT_AVAL_PLUS_AVAR:
        ax[2].plot(time, activity[AVAL, :]+activity[AVAR, :], linewidth=1.5, label=r'$F_{mc}$')
    else:
        ax[2].plot(time, activity[neuron, :], linewidth=1.5, label=r'$F_{mc}$')
    ax[2].set_ylabel('Activity')
    ax[2].set_xlabel('Time (s)')


    ax[3].plot(time, velocity, 'k', linewidth=1.5)
    ax[3].set_ylabel('Velocity')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_xticks(np.arange(0, time[-1], 60))
    ax[3].axhline(linewidth=0.5, color='k')


    ax[4].plot(time, curv)
    ax[4].set_ylabel('Curvature')
    ax[4].set_yticks([-2 * np.pi, 0, 2 * np.pi])
    ax[4].set_yticklabels([r'$-2\pi$', '0', r'$2\pi$'])
    ax[4].set_xlabel('Time (s)')
    ax[4].axhline(linewidth=0.5, color='k')

    ax[5].plot(time, deriv[neuron, :])
    ax[5].set_ylabel('dF/dt')
    ax[5].set_xlabel('Time (s)')

print("Saving tuning curve plots to PDF...")
filename = os.path.join(user_tracker.codePath(), 'figures/output/' + key + "_tuning.pdf")
pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
numFigs = plt.gcf().number + 1
for fig in xrange(1, numFigs): ## will open an empty extra figure :(
    print("Saving Figure %d of %d" % (fig, numFigs))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved: %s" % filename)








