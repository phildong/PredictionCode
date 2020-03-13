
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
#idn = 'BrainScanner20200130_110803'
idn = 'BrainScanner20200130_105254'


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


    ax1.plot(phase, activity[neuron , :], 'o', markersize=0.7, rasterized=True)
    ax1.set_xlabel('Phase (radians)')
    ax1.set_ylabel('F (motion rejected)')

    ax2.plot(neg_phase, activity[neuron , :], 'o', markersize=0.7, rasterized=True)
    ax2.set_xlabel('Negative Peak Phase (radians)')
    ax2.set_ylabel('F (motion rejected)')

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




