
################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh

print("Loading data..")

codePath = userTracker.codePath()
outputFolder = os.path.join(codePath,'figures/Debugging')

data = {}
for typ in ['AKS297.51', 'AML32', 'AML18']:
    for condition in ['moving', 'chip', 'immobilized']:  # ['moving', 'immobilized', 'chip']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path, '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path, 'Analysis/{}_{}.hdf5'.format(typ, condition))

        print("Loading " + folder + "...")
        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc)
            # store in dictionary by typ and condition
            key = '{}_{}'.format(typ, condition)
            data[key] = {}
            data[key]['dsets'] = keyList
            data[key]['input'] = dataSets
            data[key]['analysis'] = results
        except IOError:
            print typ, condition, 'not found.'
            pass
print('Done reading data.')



### CHOOSE DATASET TO PLOT
key = 'AKS297.51_moving'
idn = 'BrainScanner20200130_110803'
#idn = 'BrainScanner20200130_105254'


#key = 'AML32_moving'
#idn = 'BrainScanner20170424_105620'

### Get the relevant data.
dset = data[key]['input'][idn]
activity = dset['Neurons']['I_smooth_interp_crop_noncontig']
time = dset['Neurons']['I_Time_crop_noncontig']
numNeurons = activity.shape[0]
vel = dset['Behavior_crop_noncontig']['AngleVelocity']
comVel = dset['Behavior_crop_noncontig']['CMSVelocity']
curv = dset['Behavior_crop_noncontig']['Eigenworm3']




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if False:
    #Just compare some of the behavior
    behFig = plt.figure(figsize=[22, 9])
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

UseCOM = False
if UseCOM:
    velName = 'Eigenworm Velocity'
    velocity = vel
else:
    velName = 'COM Velocity'
    velocity = comVel




from prediction.Classifier import rectified_derivative

pos_deriv, neg_deriv = rectified_derivative(activity)

def fit_line(behavior, activity, ignore_0_activity=True, range=None):
    # examples from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq

    if ignore_0_activity:
        valid = np.nonzero(activity)
    else:
        valid = np.arange(len(activity))

    if range is 'Positive':
        range_idx = np.where(behavior > 0)
    elif range is 'Negative':
        range_idx= np.where(behavior < 0)
    else:
        range_idx = np.arange(len(behavior))

    valid = np.intersect1d(valid, range_idx)



    A = np.vstack([behavior[valid], np.ones(len(behavior[valid]))]).T
    y = activity[valid]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


#Loop through each neuron
for neuron in np.arange(numNeurons):
    fig = plt.figure(constrained_layout=True, figsize=[22, 16])
    gs = gridspec.GridSpec(ncols=6, nrows=5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax1c = fig.add_subplot(gs[0, 2], sharex=ax1)
    ax2 = fig.add_subplot(gs[0, 3])
    ax2b = fig.add_subplot(gs[0, 4], sharex=ax2)
    ax2c = fig.add_subplot(gs[0, 5], sharex=ax2)
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
    ax5 = fig.add_subplot(gs[3, :], sharex=ax3)
    ax6 = fig.add_subplot(gs[4, :], sharex=ax3)

    ax = [ax1, ax2, ax3, ax4, ax5, ax6]

    fig.suptitle(key + ' ' + idn + ' Neuron: #' + str(neuron))

    #Actually Plot
    ax[0].plot(velocity, activity[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(velocity, activity[neuron, :])
    ax[0].plot(velocity, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax[0].legend()

    ax1b.plot(velocity, pos_deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(velocity, pos_deriv[neuron, :])
    ax1b.plot(velocity, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax1b.legend()

    ax1c.plot(velocity, neg_deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(velocity, neg_deriv[neuron, :])
    ax1c.plot(velocity, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax1c.legend()

    ax[1].plot(curv, activity[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(curv, activity[neuron, :])
    ax[1].plot(curv, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax[1].legend()

    ax2b.plot(curv, pos_deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(curv, pos_deriv[neuron, :])
    ax2b.plot(curv, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax2b.legend()

    ax2c.plot(curv,  neg_deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m, c =fit_line(curv, neg_deriv[neuron, :])
    ax2c.plot(curv, m*velocity + c, 'r', label='Fit y= %.2fx+ %.2f' % (m, c), rasterized=True)
    ax2c.legend()



    ax[0].set_title(key + ' ' + idn + ' Neuron: #' + str(neuron),
                    fontsize=10)
    ax[1].set_title(key + ' ' + idn + ' Neuron: #' + str(neuron),
                    fontsize=10)

    #Add line for behavior = 0
    ax[0].axvline(linewidth=0.5, color='k')
    ax[1].axvline(linewidth=0.5, color='k')
    ax1b.axvline(linewidth=0.5, color='k')
    ax1c.axvline(linewidth=0.5, color='k')
    ax2b.axvline(linewidth=0.5, color='k')
    ax2c.axvline(linewidth=0.5, color='k')

    ax[0].set_xlabel(velName)
    ax1b.set_xlabel(velName)
    ax1c.set_xlabel(velName)
    ax[1].set_xlabel('Curvature')
    ax2b.set_xlabel('Curvature')
    ax2c.set_xlabel('Curvature')

    ax[0].set_ylabel('Fluorescence (common-noise rejected)')
    ax[1].set_ylabel('Fluorescence (common-noise rejected)')
    ax1b.set_ylabel('(+) Rectified dF/dt')
    ax1c.set_ylabel('(-) Rectified dF/dt')
    ax2b.set_ylabel('(+) Rectified dF/dt')
    ax2c.set_ylabel('(-) Rectified dF/dt')


    ax[2].plot(time, activity[neuron, :])
    ax[2].set_ylabel('Activity')
    ax[2].set_xlabel('Time (s)')


    ax[3].plot(time, velocity)
    ax[3].set_ylabel(velName)
    ax[3].set_xlabel('Time (s)')
    ax[3].axhline(linewidth=0.5, color='k')


    ax[4].plot(time, curv)
    ax[4].set_ylabel('Curvature')
    ax[4].set_xlabel('Time (s)')
    ax[4].axhline(linewidth=0.5, color='k')

    ax[5].plot(time, pos_deriv[neuron, :])
    ax[5].plot(time, neg_deriv[neuron, :])
    ax[5].set_ylabel('Rectified dF/dt')
    ax[5].set_xlabel('Time (s)')




#Later should sort by COM of the distribution.. but for now just plot


print("Saving tuning curve plots to PDF...")
filename = key + "_" + idn + "_tuning.pdf"
print(filename)
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
numFigs = plt.gcf().number + 1
for fig in xrange(1, numFigs): ## will open an empty extra figure :(
    print("Saving Figure %d of %d" % (fig, numFigs))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved.")








