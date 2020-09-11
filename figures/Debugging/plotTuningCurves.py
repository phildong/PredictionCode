
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
    behFig = plt.figure(figsize=[24, 9])
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
from inspectPerformance import calc_R2
def fit_line(behavior, activity, ignore_0_activity=False, range=None, pval=False):
    # examples from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq

    if ignore_0_activity:
        valid = np.nonzero(activity)
    else:
        valid = np.arange(len(activity))

    if range is 'Positive':
        range_idx = np.where(behavior > 0)
    elif range is 'Negative':
        range_idx = np.where(behavior < 0)
    else:
        range_idx = np.arange(len(behavior))

    valid = np.intersect1d(valid, range_idx)

    A = np.vstack([behavior[valid], np.ones(len(behavior[valid]))]).T
    y = activity[valid]
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]


    p = None

    r2 = calc_R2(y, m * behavior[valid] + c)
    rho2 = nancorrcoef(y, m * behavior[valid] + c)[0, 1] ** 2

    if pval:
        numShuffles = 5000
        shuff_y = np.matlib.repmat(y, numShuffles, 1)
        roll = np.random.randint(len(y), size=numShuffles)
        shuff_y = strided_indexing_roll(shuff_y, roll)
        shuff_y = shuff_y.T

        m_shuff, _ = np.linalg.lstsq(A, shuff_y, rcond=None)[0]
        slopes_greater_then_m = len(np.array(np.where(np.abs(m_shuff) >= np.abs(m))).ravel())
        p = np.true_divide(slopes_greater_then_m, numShuffles)
        print("p-value = %0.3f" % p)



    return m, c, p, r2, rho2


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

#Loop through each neuron
for neuron in np.arange(numNeurons):
    print("Generating plot for neuron %d" % neuron)
    fig = plt.figure(constrained_layout=True, figsize=[24, 16])
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

    #Actually Plot
    ax[0].plot(velocity, activity[neuron, :], 'o', markersize=0.7, rasterized=True)
    m_vel[neuron], c_vel[neuron], p_vel[neuron], r2_vel[neuron], rho2_vel[neuron] = fit_line(velocity, activity[neuron, :], pval=True)
    ax[0].plot(velocity, m_vel[neuron]*velocity + c_vel[neuron], 'r', label='p=%.3f, rho2=%.2f' % (p_vel[neuron], rho2_vel[neuron]), rasterized=True)
    m, c, p, r2, rho2 =fit_line(velocity, activity[neuron, :], range='Positive', pval=True)
    ax[0].plot(velocity[np.where(velocity>0)], m*velocity[np.where(velocity > 0)] + c,
              color='orange', label='(x>0)  p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    m, c, p, r2, rho2 =fit_line(velocity, activity[neuron, :], range='Negative', pval=True)
    ax[0].plot(velocity[np.where(velocity < 0)], m*velocity[np.where(velocity < 0)] + c,
              color='orange', label='(x<0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    ax[0].legend()


    #do the full derivative (not rectified)
    ax1d.plot(velocity, deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m_dfdt_vel[neuron], c_dfdt_vel[neuron], p_dfdt_vel[neuron], r2_dfdt_vel[neuron], rho2_dfdt_vel[neuron] = fit_line(velocity, deriv[neuron, :], pval=True)
    ax1d.plot(velocity, m_dfdt_vel[neuron]*velocity + c_dfdt_vel[neuron], 'r', label='p=%.3f, rho2=%.2f' % (p_dfdt_vel[neuron], rho2_dfdt_vel[neuron]), rasterized=True)
    m, c, p, r2, rho2 =fit_line(velocity, deriv[neuron, :], range='Positive', pval=True)
    ax1d.plot(velocity[np.where(velocity>0)], m*velocity[np.where(velocity>0)] + c,
              color='orange', label='(x>0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    m, c, p, r2, rho2 =fit_line(velocity, deriv[neuron, :], range='Negative', pval=True)
    ax1d.plot(velocity[np.where(velocity < 0)], m*velocity[np.where(velocity < 0)] + c,
              color='orange', label='(x<0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    ax1d.legend()

    ax[1].plot(curv, activity[neuron, :], 'o', markersize=0.7, rasterized=True)
    m_curv[neuron], c_curv[neuron], p_curv[neuron], r2_curv[neuron], rho2_curv[neuron] = fit_line(curv, activity[neuron, :], pval=True)
    ax[1].plot(curv, m_curv[neuron]*curv + c_curv[neuron], 'r', label='p=%.3f, rho2=%.2f' % (p_curv[neuron], rho2_curv[neuron]), rasterized=True)
    m, c, p, r2, rho2 =fit_line(curv, activity[neuron, :], range='Positive', pval=True)
    ax[1].plot(curv[np.where(curv>0)], m*curv[np.where(curv > 0)] + c,
              color='orange', label='(x>0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    m, c, p, r2, rho2 =fit_line(curv, activity[neuron, :], range='Negative', pval=True)
    ax[1].plot(curv[np.where(curv < 0)], m*curv[np.where(curv < 0)] + c,
              color='orange', label='(x<0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    ax[1].legend()


    ax2d.plot(curv,  deriv[neuron, :], 'o', markersize=0.7, rasterized=True)
    m_dfdt_curv[neuron], c_dfdt_curv[neuron], p_dfdt_curv[neuron], r2_dfdt_curv[neuron], rho2_dfdt_curv[neuron] = fit_line(curv, deriv[neuron, :], pval=True)
    ax2d.plot(curv, m_dfdt_curv[neuron]*velocity + c_dfdt_curv[neuron], 'r', label='p=%.3f, rho2=%.2f' % (p_dfdt_curv[neuron], rho2_dfdt_curv[neuron]), rasterized=True)
    m, c, p, r2, rho2 = fit_line(curv, deriv[neuron, :], range='Positive', pval=True)
    ax2d.plot(curv[np.where(curv > 0)], m * curv[np.where(curv > 0)] + c,
               color='orange', label='(x>0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    m, c, p, r2, rho2 = fit_line(curv, deriv[neuron, :], range='Negative', pval=True)
    ax2d.plot(curv[np.where(curv < 0)], m * curv[np.where(curv < 0)] + c,
               color='orange', label='(x<0) p=%.3f, rho2=%.2f' % (p, rho2), rasterized=True)
    ax2d.legend()


    ax[0].set_title(key + ' ' + idn + ' Neuron: #' + str(neuron),
                    fontsize=10)
    ax[1].set_title(key + ' ' + idn + ' Neuron: #' + str(neuron),
                    fontsize=10)

    #Add line for behavior = 0
    ax[0].axvline(linewidth=0.5, color='k')
    ax[1].axvline(linewidth=0.5, color='k')
    ax1d.axvline(linewidth=0.5, color='k')
    ax2d.axvline(linewidth=0.5, color='k')


    ax[0].set_xlabel(velName)
    ax1d.set_xlabel(velName)
    ax[1].set_xlabel('Curvature')
    ax2d.set_xlabel('Curvature')

    ax[0].set_ylabel('Fluorescence (common-noise rejected)')
    ax[1].set_ylabel('Fluorescence (common-noise rejected)')
    ax1d.set_ylabel('dF/dt')
    ax2d.set_ylabel('dF/dt')

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

    ax[5].plot(time, deriv[neuron, :])
    ax[5].set_ylabel('dF/dt')
    ax[5].set_xlabel('Time (s)')

    import prediction.provenance as prov
    prov.stamp(ax[5], .55, .15)


if False:
    #close all figures (temporarily for debugging purposes)
    numFigs = plt.gcf().number + 1
    for fig in xrange(1, numFigs): ## will open an empty extra figure :(
        print("Closing Figure %d of %d" % (fig, numFigs))
        plt.close(fig)


#Find the neurons with the largest magnitude slopes
neurons_tune2_vel = np.flip(np.argsort(np.abs(m_vel)))
neurons_w_pos_vel_tuning = neurons_tune2_vel[np.where(m_vel[neurons_tune2_vel] > 0)]
neurons_w_neg_vel_tuning = neurons_tune2_vel[np.where(m_vel[neurons_tune2_vel] < 0)]

#plot abs value of the slopes
vel_fig = plt.figure(figsize=[24, 9])
gs = gridspec.GridSpec(ncols=2, nrows=3, figure=vel_fig)
vax = vel_fig.add_subplot(gs[0, 0])
vax.plot(np.arange(len(neurons_w_pos_vel_tuning)), m_vel[neurons_w_pos_vel_tuning],'-o')
vax.set_title('Magnitude of velocity tuning for positive tuned neurons [activity = m * velocity + c]')
vax.set_ylabel('Slope (m)')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_pos_vel_tuning)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_vel_tuning)))
vax.text(10, 30000, 'This shows the abs value of the slope, m. But it is biased towards neurons whose intensity changes over a large range.')

vax = vel_fig.add_subplot(gs[0, 1])
vax.plot(np.arange(len(neurons_w_neg_vel_tuning)), m_vel[neurons_w_neg_vel_tuning],'-o')
vax.set_title('Magnitude of velocity tuning for negative tuned neurons [activity = m * velocity + c]')
vax.set_ylabel('Slope (m)')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_neg_vel_tuning)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_vel_tuning)))
vax.text(10, 30000, 'This shows the abs value of the slope, m. But it is biased towards neurons whose intensity changes over a large range.')




# Plot again, but this tome plot the fraction of the range explained by the linear fit, for each neuron
fit_activity_range = m_vel*(np.max(velocity) - np.min(velocity))
activity_range = np.max(activity, axis=1) - np.min(activity, axis=1)
frac_range = np.divide(fit_activity_range, activity_range)

#find the neurons with the largest frac range expalined by the fit
neurons_tune2_vel_range = np.flip(np.argsort(np.abs(frac_range)))
neurons_w_pos_vel_r_tuning = neurons_tune2_vel_range[np.where(frac_range[neurons_tune2_vel_range] > 0)]
neurons_w_neg_vel_r_tuning = neurons_tune2_vel_range[np.where(frac_range[neurons_tune2_vel_range] < 0)]

vax = vel_fig.add_subplot(gs[1, 0])
vax.plot(np.arange(len(neurons_w_pos_vel_r_tuning)), frac_range[neurons_w_pos_vel_r_tuning],'-o')
vax.set_title('Fraction of activity range fit')
vax.set_ylabel('Fraction of activity range covered by fit')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_pos_vel_r_tuning)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_vel_r_tuning)))
vax.text(20, .5, '(range of fit line) / (range of neural actiivty) ')

vax = vel_fig.add_subplot(gs[1, 1])
vax.plot(np.arange(len(neurons_w_neg_vel_r_tuning)), frac_range[neurons_w_neg_vel_r_tuning],'-o')
vax.set_title('Fraction of activity range fit')
vax.set_ylabel('Fraction of activity range covered by fit')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_neg_vel_r_tuning)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_vel_r_tuning)))
vax.text(20, .5, '(range of fit line) / (range of neural actiivty) ')



# Plot again, but this tome plot the p-value of the slope

#find the neurons with the smallest p-value
neurons_tune2_vel_p = np.argsort(p_vel)
neurons_w_pos_vel_tuning_p = neurons_tune2_vel_p[np.where(m_vel[neurons_tune2_vel_p] > 0)]
neurons_w_neg_vel_tuning_p = neurons_tune2_vel_p[np.where(m_vel[neurons_tune2_vel_p] < 0)]

vax = vel_fig.add_subplot(gs[2, 0])
vax.plot(np.arange(len(neurons_w_pos_vel_tuning_p)), p_vel[neurons_w_pos_vel_tuning_p],'-o')
vax.set_title('p-value for  positive slope tp velocity ')
vax.set_ylabel('p-value')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_pos_vel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_vel_tuning_p)))

vax = vel_fig.add_subplot(gs[2, 1])
vax.plot(np.arange(len(neurons_w_neg_vel_tuning_p)), p_vel[neurons_w_neg_vel_tuning_p],'-o')
vax.set_title('p-value for negative slope to velocity')
vax.set_ylabel('p-value')
vax.set_xlabel('Neuron')
vax.set_xticks(np.arange(len(neurons_w_neg_vel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_vel_tuning_p)))

vel_fig.tight_layout()

####
### Setup a new figure that just compares p-values
###
# but does the comparison for the overall velocity, as well as the rectified derivitive velocities.
vel_fig2 = plt.figure(figsize=[24, 9])
vel_fig2.suptitle(key + ' ' + idn + ' Neuron: #' + str(neuron))
gs2 = gridspec.GridSpec(ncols=2, nrows=3, figure=vel_fig2)


#find the neurons with the smallest p-value
neurons_tune2_vel_p = np.argsort(p_vel)
neurons_w_pos_vel_tuning_p = neurons_tune2_vel_p[np.where(m_vel[neurons_tune2_vel_p] > 0)]
neurons_w_neg_vel_tuning_p = neurons_tune2_vel_p[np.where(m_vel[neurons_tune2_vel_p] < 0)]

def add_y_logscale_and_labels(ax):
    ax.set_yscale('log')
    ax.axhline(0.05, linestyle='--', color='r')
    ax.axhline(0.01, linestyle='-.', color='orange')
    ax.set_ylabel('p-value')
    ax.set_xlabel('Neuron')
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    return

vax = vel_fig2.add_subplot(gs2[0, 0])
vax.plot(np.arange(len(neurons_w_pos_vel_tuning_p)), p_vel[neurons_w_pos_vel_tuning_p], '-o')
vax.set_title('p-value for positive slope to velocity')
vax.set_xticks(np.arange(len(neurons_w_pos_vel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_vel_tuning_p)))
add_y_logscale_and_labels(vax)


vax = vel_fig2.add_subplot(gs2[0, 1])
vax.plot(np.arange(len(neurons_w_neg_vel_tuning_p)), p_vel[neurons_w_neg_vel_tuning_p], '-o')
vax.set_title('p-value for negative slope to velocity')
vax.set_xticks(np.arange(len(neurons_w_neg_vel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_vel_tuning_p)))
add_y_logscale_and_labels(vax)


#find the neurons with the smallest p-values for the positive rectified derivitive
neurons_tune2_prvel_p = np.argsort(p_p_vel)
neurons_w_pos_prvel_tuning_p = neurons_tune2_prvel_p[np.where(m_p_vel[neurons_tune2_prvel_p] > 0)]
neurons_w_neg_prvel_tuning_p = neurons_tune2_prvel_p[np.where(m_p_vel[neurons_tune2_prvel_p] < 0)]

vax = vel_fig2.add_subplot(gs2[1, 0])
vax.plot(np.arange(len(neurons_w_pos_prvel_tuning_p)), p_p_vel[neurons_w_pos_prvel_tuning_p],'-o')
vax.set_title('p-value for positive slope  to positive rectified dv/dt')
vax.set_xticks(np.arange(len(neurons_w_pos_prvel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_prvel_tuning_p)))
add_y_logscale_and_labels(vax)

vax = vel_fig2.add_subplot(gs2[1, 1])
vax.plot(np.arange(len(neurons_w_neg_prvel_tuning_p)), p_p_vel[neurons_w_neg_prvel_tuning_p],'-o')
vax.set_title('p-value for negative slope to positive rectified dv/dt')
vax.set_xticks(np.arange(len(neurons_w_neg_prvel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_prvel_tuning_p)))
add_y_logscale_and_labels(vax)

#find the neurons with the smallest p-values for the negative rectified derivitive
neurons_tune2_nrvel_p = np.argsort(p_n_vel)
neurons_w_pos_nrvel_tuning_p = neurons_tune2_nrvel_p[np.where(m_n_vel[neurons_tune2_nrvel_p] > 0)]
neurons_w_neg_nrvel_tuning_p = neurons_tune2_nrvel_p[np.where(m_n_vel[neurons_tune2_nrvel_p] < 0)]

vax = vel_fig2.add_subplot(gs2[2, 0])
vax.plot(np.arange(len(neurons_w_pos_nrvel_tuning_p)), p_n_vel[neurons_w_pos_nrvel_tuning_p],'-o')
vax.set_title('p-value for positive slope to negative rectified dv/dt')
vax.set_xticks(np.arange(len(neurons_w_pos_nrvel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_pos_nrvel_tuning_p)))
add_y_logscale_and_labels(vax)

vax = vel_fig2.add_subplot(gs2[2, 1])
vax.plot(np.arange(len(neurons_w_neg_nrvel_tuning_p)), p_n_vel[neurons_w_neg_nrvel_tuning_p], '-o')
vax.set_title('p-value for negative slope to negative rectified dv/dt')
vax.set_xticks(np.arange(len(neurons_w_neg_nrvel_tuning_p)))
vax.set_xticklabels(np.array(map(str, neurons_w_neg_nrvel_tuning_p)))
add_y_logscale_and_labels(vax)

import prediction.provenance as prov
prov.stamp(vax, .55, .15)
vel_fig2.tight_layout()



##### Plot rho2 and p-values together (STILL VELOCITY)
####
###
# but do the comparison for the overall velocity
vel_fig3 = plt.figure(figsize=[24, 9])
vel_fig3.suptitle(key + ' ' + idn )
gs3 = gridspec.GridSpec(ncols=2, nrows=1, figure=vel_fig3)

max_rho2 = np.max(np.array([rho2_vel, rho2_n_vel, rho2_p_vel, rho2_dfdt_vel]).flatten())

vax = vel_fig3.add_subplot(gs3[0, 0])
vax.plot(p_vel[np.where(m_vel > 0)], rho2_vel[np.where(m_vel > 0)], 'bo', label="positive slope")
vax.plot(p_vel[np.where(m_vel <= 0)], rho2_vel[np.where(m_vel <= 0)], 'ro', label="negative slope")
vax.set_title('slope to velocity')
vax.set_xlabel('p-value')
vax.set_ylabel('correlation coefficient rho2')
vax.set_xscale('log')
vax.set_ylim(0, max_rho2*1.2)
vax.axvline(0.05, linestyle='--', color='r')
vax.axvline(0.01, linestyle='-.', color='orange')
vax.axhline(0.15, linestyle='--', color='magenta')
for i in np.arange(len(p_vel)):
    vax.annotate(i, (p_vel[i], rho2_vel[i]))
vax.legend()


vax = vel_fig3.add_subplot(gs3[0, 1])
vax.plot(p_dfdt_vel[np.where(m_dfdt_vel > 0)], rho2_dfdt_vel[np.where(m_dfdt_vel > 0)], 'bo', label="positive slope")
vax.plot(p_dfdt_vel[np.where(m_dfdt_vel <= 0)], rho2_dfdt_vel[np.where(m_dfdt_vel <= 0)], 'ro', label="negative slope")
vax.set_title('slope to  df/dt')
vax.set_xlabel('p-value')
vax.set_ylabel('correlation coefficient rho2')
vax.set_xscale('log')
vax.axvline(0.05, linestyle='--', color='r')
vax.axvline(0.01, linestyle='-.', color='orange')
vax.axhline(0.15, linestyle='--', color='magenta')
for i in np.arange(len(p_vel)):
    vax.annotate(i, (p_dfdt_vel[i], rho2_dfdt_vel[i]))
vax.set_ylim(0, max_rho2*1.2)
vax.legend()


import prediction.provenance as prov
prov.stamp(vax, .55, .15)
vel_fig3.tight_layout()



##### Plot rho2 and p-values together (NOW CURVATURE)
####
###
curv_fig4 = plt.figure(figsize=[24, 9])
curv_fig4.suptitle(key + ' ' + idn)
gs4 = gridspec.GridSpec(ncols=2, nrows=1, figure=curv_fig4)

max_rho2 = np.max(np.array([rho2_curv, rho2_n_curv, rho2_p_curv]).flatten())

cax = curv_fig4.add_subplot(gs4[0, 0])
cax.plot(p_curv[np.where(m_curv > 0)], rho2_curv[np.where(m_curv > 0)], 'bo', label="positive slope")
cax.plot(p_curv[np.where(m_curv <= 0)], rho2_curv[np.where(m_curv <= 0)], 'ro', label="negative slope")
cax.set_title('slope to curvature')
cax.set_xlabel('p-value')
cax.set_ylabel('correlation coefficient rho2')
cax.set_xscale('log')
cax.set_ylim(0, max_rho2*1.2)
cax.axvline(0.05, linestyle='--', color='r')
cax.axvline(0.01, linestyle='-.', color='orange')
cax.axhline(0.15, linestyle='--', color='magenta')
for i in np.arange(len(p_curv)):
    cax.annotate(i, (p_curv[i], rho2_curv[i]))
cax.legend()

cax = curv_fig4.add_subplot(gs4[0, 1])
cax.plot(p_dfdt_curv[np.where(m_dfdt_curv > 0)], rho2_dfdt_curv[np.where(m_dfdt_curv > 0)], 'bo', label="positive slope")
cax.plot(p_dfdt_curv[np.where(m_dfdt_curv <= 0)], rho2_dfdt_curv[np.where(m_dfdt_curv <= 0)], 'ro', label="negative slope")
cax.set_title('slope of df/dft to curvature')
cax.set_xlabel('p-value')
cax.set_ylabel('correlation coefficient rho2')
cax.set_xscale('log')
cax.set_ylim(0, max_rho2*1.2)
cax.axvline(0.05, linestyle='--', color='r')
cax.axvline(0.01, linestyle='-.', color='orange')
cax.axhline(0.15, linestyle='--', color='magenta')
for i in np.arange(len(p_dfdt_curv)):
    cax.annotate(i, (p_dfdt_curv[i], rho2_dfdt_curv[i]))
cax.legend()





import prediction.provenance as prov
prov.stamp(cax, .55, .15)
curv_fig4.tight_layout()


if UseEigVol:
    velStyle = "eig"
else:
    velStyle = "com"

print("Saving tuning curve plots to PDF...")
filename = key + "_" + idn + "_tuning_" + velStyle + ".pdf"
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








