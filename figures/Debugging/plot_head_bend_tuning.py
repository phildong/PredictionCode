
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


##IMPORTANT:
#We should just redo CL to be like FULL and re-run metarunanalysis..
# but until then.. here is my quick and dirty trick


#The centerline variable could concievably go beyond the last non BFP frame
# We can tell by only including good volumes less than the lastNonBFP frame



#dset['CL'] #Already is cropped to only "good volumes
            #But potentially problematic because good volumes coul din principle
            #include the BFP timepoints that are already exclucded in "Full"


centerlines = np.array(dset['CLFull'])

time = np.array(dset['Neurons']['TimeFull'])

activity = np.array(dset['Neurons']['I_smooth'])
numNeurons = activity.shape[0]






import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calc_head_angle(centerlines):
    nose_start = 0
    nose_to_neck = 10
    neck_end = 20

    nose_vec = np.diff(centerlines[:, [nose_start, nose_to_neck], :], axis=1)
    neck_vec = np.diff(centerlines[:, [nose_to_neck, neck_end], :], axis=1)

    nose_orientation = np.zeros(centerlines.shape[0])
    neck_orientation = np.zeros(centerlines.shape[0])

    for timept in np.arange(centerlines.shape[0]):
        nose_orientation[timept] = np.arctan2(nose_vec[timept,:,0], nose_vec[timept,:,1])
        neck_orientation[timept] = np.arctan2(neck_vec[timept,:,0], neck_vec[timept,:,1])

    head_angle = nose_orientation - neck_orientation
    return head_angle


head_angle = np.unwrap(calc_head_angle(centerlines))
sigma = 3
smooth_head_angle = dh.gauss_filterNaN(head_angle, sigma)


if idn == 'BrainScanner20200130_105254':
    prominence = 0.2
else:
    prominence = 0.4

## Find peaks of head swings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
prominence=0.4
peaks, _ = find_peaks(smooth_head_angle, height=-.3, prominence=prominence)
neg_peaks, _ = find_peaks(smooth_head_angle*-1, height=-.3, prominence=prominence)


def find_phase(peaks, time):
    #Use interpolation

    cum_phase_pks = 2*np.pi * np.arange(peaks.shape[0])

    from scipy import interpolate
    f = interpolate.interp1d(time[peaks], cum_phase_pks, fill_value="extrapolate")

    cum_phase = f(time)

    phase = np.mod(cum_phase, 2*np.pi)

    return phase

phase = find_phase(peaks, time)
neg_phase = find_phase(neg_peaks, time)


#functions to fit a cosine wave
from skimage.util.shape import view_as_windows as viewW
def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    # This function will roll each row of a matrix a, a an amount specified by r.
    # I got it here: https://stackoverflow.com/a/51613442/200688
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), (n-r) % n, 0]



def fit_cos_wave(activity, phase):
    """"
    Fit a cosine wave to the neural activity tuning as a function of head bending phase.
    y = A * cos( k*x + phi) + c
    where k is fixed to be 1

    """


    #Mean subtract.
    offset = np.nanmean(activity)
    activity = activity - offset

    #Scale activity to something around one
    scaleFactor = np.nanmax(activity)
    activity = activity / scaleFactor

    #Now activity shoudl be mean zero amplitude 1

    from scipy.optimize import curve_fit

    # set up some bounds on our cosine wave fitting parameters, y=A*sin(kx - phi) + c
    c_guess = np.nanmean(activity)
    c_max = np.nanmax(activity)
    c_min = np.nanmin(activity)

    A_guess = (np.nanmax(activity) - np.nanmin(activity)) / 2
    A_max = np.nanmax(activity) - np.nanmin(activity)
    A_min = 0

    phi_guess = np.pi
    phi_min = 0
    phi_max = 2 * np.pi

    # format A, phi, c
    bounds = ([A_min, phi_min, c_min],  # lower bounds
              [A_max, phi_max, c_max])  # upper bound

    popt_guess = [A_guess, phi_guess, c_guess]
    # Identify just the data that is not a NaN
    nonNaNs = np.logical_not(np.isnan(activity))
    popt, pcov = curve_fit(cos_wave, phase[nonNaNs], activity[nonNaNs], p0=popt_guess, bounds=bounds)


    ## Now we want to inspect our fit, find values that are clear outliers, and refit while excluding those
    residual = activity - cos_wave(phase, *popt) #its ok to have nans here
    nSigmas = 3  # we want to exclude points that are three standard deviations away from the fit

    # Make a new mask that excludes the outliers
    excOutliers = np.copy(nonNaNs)
    excOutliers[(np.abs(residual) > (nSigmas * np.nanstd(residual)))] = False

    # Refit excluding the outliers, use the previous fit as initial guess
    # note we relax the bounds here a bit
    try:
        popt, pcov = curve_fit(cos_wave, phase[excOutliers], activity[excOutliers], p0=popt, bounds=bounds)
    except:
        popt, pcov = curve_fit(cos_wave, phase[excOutliers], activity[excOutliers], p0=popt)


    #rescale the amplitude, a, back to full size
    popt[0] = scaleFactor*popt[0]

    #rescale the c parameter
    popt[2] = scaleFactor *popt[2] + offset
    return popt, pcov


def cos_wave(x, A, phi, c, k=1):
    # type: (xVals, A, phi, c) -> yVals
    return A * np.cos(k * x - phi) + c





import numpy.matlib
from inspectPerformance import calc_R2
def check_cosine_tuning(phase, activity, pval=False):
    # examples from https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq



    popt, pcov = fit_cos_wave(activity, phase)

    A = popt[0]
    phi = popt[1]
    c = popt[2]

    p = None

    r2 = calc_R2(activity, cos_wave(phase, A, phi, c))


    print("Still need to implement shuffle")
    if False:
        numShuffles = 5000
        shuff_y = np.matlib.repmat(y, numShuffles, 1)
        roll = np.random.randint(len(y), size=numShuffles)
        shuff_y = strided_indexing_roll(shuff_y, roll)
        shuff_y = shuff_y.T

        m_shuff, _ = np.linalg.lstsq(A, shuff_y, rcond=None)[0]
        slopes_greater_then_m = len(np.array(np.where(np.abs(m_shuff) >= np.abs(m))).ravel())
        p = np.true_divide(slopes_greater_then_m, numShuffles)
        print("p-value = %0.3f" % p)


    return A, phi, c, p, r2

r2_phase = np.zeros(numNeurons)
r2_negphase = r2_phase.copy()

phi = np.zeros(numNeurons)
phi_neg = np.zeros(numNeurons)

#Loop through each neuron
for neuron in np.arange(numNeurons):

    fig = plt.figure(constrained_layout=True, figsize=[22, 10])
    fig.suptitle(key + ' ' + idn + ' Neuron: #' + str(neuron))

    gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3)


    ax0.plot(head_angle, activity[neuron , :], 'o', markersize=0.7, rasterized=True)
    ax0.set_xlabel('Head Bend (radians)')
    ax0.set_ylabel('F (motion rejected)')

    theta = np.arange(0,1,.01)*2*np.pi

    ax1.plot(phase, activity[neuron , :], 'o', markersize=0.7, rasterized=True)
    A, phi[neuron], c, _, r2_phase[neuron] = check_cosine_tuning(phase, activity[neuron, :])
    ax1.plot(theta, cos_wave(theta, A, phi[neuron], c), label="r2=%.2f" % r2_phase[neuron])
    ax1.set_xlabel('Phase (radians)')
    ax1.set_ylabel('F (motion rejected)')
    ax1.legend()

    ax2.plot(neg_phase, activity[neuron , :], 'o', markersize=0.7, rasterized=True)
    A, phi_neg[neuron], c, _, r2_negphase[neuron] = check_cosine_tuning(neg_phase, activity[neuron, :])
    ax2.plot(theta, cos_wave(theta, A, phi_neg[neuron], c), label="r2=%.2f" % r2_negphase[neuron])
    ax2.set_xlabel('Negative Peak Phase (radians)')
    ax2.set_ylabel('F (motion rejected)')
    ax2.legend()

    ax3.plot(time, activity[neuron, :])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('F (motion rejected)')
    ax3.set_xlim(time[np.round(activity.shape[1] / 4)], time[3 * np.round(activity.shape[1] / 4)])

    ax4.plot(time, head_angle)
    ax4.plot(time[peaks], head_angle[peaks], "x", markersize=12)
    ax4.plot(time[neg_peaks], head_angle[neg_peaks], "o", markersize=12)
    ax4.axhline(color='k')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Head Bend (radians)')

    import prediction.provenance as prov
    prov.stamp(ax4, .55, .15)



# Plot r2 as a function of phi for calcualting phase based on the positive peak
fig_r2_phi = plt.figure(figsize=[24, 9])
fig_r2_phi.suptitle(key + ' ' + idn)
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig_r2_phi)

max_r2 = np.max(np.array([r2_phase, r2_negphase]).flatten())

ax_r21 = fig_r2_phi.add_subplot(gs[0, 0])
ax_r21.plot(phi, r2_phase, 'o', label="r2")
ax_r21.set_title('Cosine Wave Goodness of Fit')
ax_r21.set_xlabel('Head Bend Phase')
ax_r21.set_ylabel('coefficient of determination R2')
ax_r21.set_ylim(0, max_r2)
for i in np.arange(len(r2_phase)):
    ax_r21.annotate(i, (phi[i], r2_phase[i]))
ax_r21.legend()


ax_r22 = fig_r2_phi.add_subplot(gs[0, 1])
ax_r22.plot(phi_neg, r2_negphase, 'o', label="r2  (phase to negative headbend)")
ax_r22.set_title('Cosine Wave Goodness of Fit, to phase calculated on negative bends')
ax_r22.set_xlabel('Head Bend Phase (calculated on negative bends)')
ax_r22.set_ylabel('coefficient of determination R2')
ax_r22.set_ylim(0, max_r2)
for i in np.arange(len(r2_negphase)):
    ax_r22.annotate(i, (phi_neg[i], r2_negphase[i]))
ax_r22.legend()






print("Saving head angle plots to PDF...")
filename = key + "_" + idn + "head_tuning.pdf"
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




