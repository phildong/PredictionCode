
################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh

COMPARE_MOTION_CORRECTION = False
PLOT_AVAL_PLUS_AVAR = False
print("Loading data..")

codePath = userTracker.codePath()
outputFolder = os.path.join(codePath,'figures/Debugging')

data = {}
for typ_cond in ['AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path, '{0}/{0}_datasets.txt'.format(typ_cond))

    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder)
    keyList = np.sort(dataSets.keys())
    data[typ_cond] = {}
    data[typ_cond]['dsets'] = keyList
    data[typ_cond]['input'] = dataSets



### CHOOSE DATASET TO PLOT
key = 'AKS297.51_moving'
idn = 'BrainScanner20200130_110803'
#key = 'AML18_moving'
#idn = 'BrainScanner20200204_102136'

#idn = 'BrainScanner20200130_105254'
#key = 'AML32_moving'
#idn = 'BrainScanner20170424_105620'

### Get the relevant data.
dset = data[key]['input'][idn]
activity = dset['Neurons']['I_smooth_interp_crop_noncontig']
raw = dset['Neurons']['G_smooth_interp_crop_noncontig']
R = dset['Neurons']['R_smooth_interp_crop_noncontig']
time = dset['Neurons']['I_Time_crop_noncontig']
numNeurons = activity.shape[0]
comVel = dset['Behavior_crop_noncontig']['CMSVelocity']
curv = dset['Behavior_crop_noncontig']['Curvature']




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if False:
    #Just compare some of the behavior
    behFig = plt.figure(figsize=[12, 9])
    behGs = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig)
    bax1 = behFig.add_subplot(behGs[0,0])
    bax2 = behFig.add_subplot(behGs[1,0], sharex=bax1)
    bax3 = behFig.add_subplot(behGs[2,0], sharex=bax1)

    neuron =14
    bax1.plot(activity[neuron, :])
    bax1.set_title('Activity of AVA, #%d' % neuron)

    bax2.plot(vel, label='eigenworm velocity')
    bax2.set_title('Eigenworm Based Velocity'
                   )
    bax3.plot(comVel, label='incorrectly calibrated com velocity')
    bax3.set_title('COM Velocity')


    plt.figure(figsize=[10,10])
    plt.plot(vel, comVel)

UseEigVol = False
if UseEigVol:
    velName = 'Eigenworm Velocity'
    velocity = vel
else:
    velName = 'COM Velocity'
    velocity = comVel


from prediction.Classifier import rectified_derivative
pos_deriv, neg_deriv, deriv = rectified_derivative(activity)

from skimage.util.shape import view_as_windows as viewW

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    # This function will roll each row of a matrix a, a an amount specified by r.
    # I got it here: https://stackoverflow.com/a/51613442/200688
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]


import numpy.ma as ma
def nancorrcoef(A, B):
    a = ma.masked_invalid(A)
    b = ma.masked_invalid(B)

    msk = (~a.mask & ~b.mask)

    return ma.corrcoef(a[msk], b[msk])

import numpy.matlib


m_vel = np.zeros((1, numNeurons)).flatten()
m_p_vel = np.copy(m_vel) # m positive rectified
m_n_vel = np.copy(m_vel) # m negative rectified
m_dfdt_vel = np.copy(m_vel) # derivative

m_curv = np.zeros((1, numNeurons)).flatten()
m_p_curv = np.copy(m_vel) # m positive rectified
m_n_curv = np.copy(m_vel) # m negative rectified
m_dfdt_curv = np.copy(m_vel) # m slope to derivative

c_vel = np.zeros((1, numNeurons)).flatten()
c_p_vel = np.copy(c_vel) # c positive rectified
c_n_vel = np.copy(c_vel) # c negative rectified
c_dfdt_vel = np.copy(c_vel) # c offset for derivative

c_curv = np.zeros((1, numNeurons)).flatten()
c_p_curv = np.copy(c_vel) # c positive rectified
c_n_curv = np.copy(c_vel) # c negative rectified
c_dfdt_curv = np.copy(c_vel) # c offset for derivative

#p-values on the slope
p_vel = np.zeros((1, numNeurons)).flatten()
p_p_vel = np.copy(c_vel) #  positive rectified
p_n_vel = np.copy(c_vel) #  negative rectified
p_dfdt_vel = np.copy(c_vel) #  df/dt


p_curv = np.zeros((1, numNeurons)).flatten()
p_p_curv = np.copy(c_vel) #  positive rectified
p_n_curv = np.copy(c_vel) #  negative rectified
p_dfdt_curv = np.copy(c_vel) #  df/dt

r2_vel = np.copy(p_vel)
r2_p_vel = np.copy(p_vel)
r2_n_vel = np.copy(p_vel)
r2_dfdt_vel = np.copy(p_vel) #  df/dt

rho2_vel = np.copy(p_vel)
rho2_p_vel = np.copy(p_vel)
rho2_n_vel = np.copy(p_vel)
rho2_dfdt_vel = np.copy(p_vel) #  df/dt


r2_curv = np.copy(p_vel)
r2_p_curv = np.copy(p_vel)
r2_n_curv = np.copy(p_vel)
r2_dfdt_curv = np.copy(p_vel) # df/dt

rho2_curv = np.copy(p_vel)
rho2_p_curv = np.copy(p_vel)
rho2_n_curv = np.copy(p_vel)
rho2_dfdt_curv = np.copy(p_vel) # df/dt

AVAR = 32
AVAL = 15
#Loop through each neuron
for neuron in np.array([AVAR, AVAL]): #np.arange(numNeurons):
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
    if COMPARE_MOTION_CORRECTION:
        ax[2].plot(time, raw[neuron, :]-np.nanmean(raw[neuron, :]), color='green', linewidth=1.5, label=r'$F_{GCaMP}$')
        #ax[2].plot(time, R[neuron, :]-np.nanmean(R[neuron, :]), color='red', linewidth=1.5, label=r'$F_{RFP}$')
        ax[2].legend()
    ax[2].set_ylabel('Activity')
    ax[2].set_xlabel('Time (s)')


    ax[3].plot(time, velocity, 'k', linewidth=1.5)
    ax[3].set_ylabel(velName)
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




if UseEigVol:
    velStyle = "eig"
else:
    velStyle = "com"

print("Saving tuning curve plots to PDF...")
filename = 'generatedFigs/' + key + "_" + idn + "_tuning_" + velStyle + ".pdf"
print(filename)
import matplotlib.backends.backend_pdf
import prediction.provenance as prov
pdf = matplotlib.backends.backend_pdf.PdfPages(filename, metadata=prov.pdf_metadata(__file__))
numFigs = plt.gcf().number + 1
for fig in xrange(1, numFigs): ## will open an empty extra figure :(
    print("Saving Figure %d of %d" % (fig, numFigs))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved.", filename)








