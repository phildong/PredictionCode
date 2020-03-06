
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


anti_AVA_like = [11, 24, 74, 30,85,  88]

#Just compare some of the behavior
behFig2 = plt.figure(figsize=[24, 12])
behGs2 = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig2
                           )
b2ax1 = behFig2.add_subplot(behGs2[0,0])
b2ax2 = behFig2.add_subplot(behGs2[1,0], sharex=b2ax1)
b2ax3 = behFig2.add_subplot(behGs2[2,0], sharex=b2ax1)

for neuron in anti_AVA_like:
    b2ax1.plot(time, activity[neuron, :], label='Neuron %d' % neuron)
b2ax1.set_ylabel('Motion-corrected fluorescence intensity F')
b2ax1.set_xlabel('Time (s)')
b2ax1.legend()

b2ax1.set_title('Activity of Select Neurons')

b2ax2.plot(time, vel, label='eigenworm velocity')
b2ax2.set_title('Eigenworm Based Velocity')
b2ax2.axhline()

b2ax3.plot(time, comVel*50, label=' com velocity')
b2ax3.set_title('COM Velocity')
b2ax3.set_ylabel('Velocity (mm/s)')
b2ax3.axhline()

import prediction.provenance as prov
prov.stamp(b2ax3, .55, .15)

plt.show()

from prediction.Classifier import rectified_derivative
pos_deriv, neg_deriv = rectified_derivative(activity)

#Just compare some of the behavior
behFig4 = plt.figure(figsize=[24, 12])
behGs4 = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig4
                           )
b4ax1 = behFig4.add_subplot(behGs4[0,0])
b4ax2 = behFig4.add_subplot(behGs4[1,0], sharex=b4ax1)
b4ax3 = behFig4.add_subplot(behGs4[2,0], sharex=b4ax1)

for neuron in anti_AVA_like:
    b4ax1.plot(time, neg_deriv[neuron, :], label='Neuron %d' % neuron)
b4ax1.set_ylabel('(-) Rectified dF/dt')
b4ax1.set_xlabel('Time (s)')
b4ax1.legend()

b4ax1.set_title('Activity of Select Neurons')

b4ax2.plot(time, vel, label='eigenworm velocity')
b4ax2.set_title('Eigenworm Based Velocity')
b4ax2.axhline()

b4ax3.plot(time, comVel*50, label=' com velocity')
b4ax3.set_title('COM Velocity')
b4ax3.set_ylabel('Velocity (mm/s)')
b4ax3.axhline()

import prediction.provenance as prov
prov.stamp(b4ax3, .55, .15)


plt.show()


AVA_like_neurons = np.array([53, 56, 59, 1, 33, 49])

#Just compare some of the behavior
behFig = plt.figure(figsize=[24, 12])
behGs = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig)
bax1 = behFig.add_subplot(behGs[0,0])
bax2 = behFig.add_subplot(behGs[1,0], sharex=bax1)
bax3 = behFig.add_subplot(behGs[2,0], sharex=bax1)

for neuron in AVA_like_neurons:
    bax1.plot(time, activity[neuron, :], label='Neuron %d (AVA)' % neuron)

bax1.set_ylabel('Motion-corrected fluorescence intensity F')
bax1.set_xlabel('Time (s)')
bax1.legend()
bax1.set_title('Activity of AVA-Like Neurons')

bax2.plot(time, vel, label='eigenworm velocity')
bax2.set_title('Eigenworm Based Velocity')
bax2.axhline()

bax3.plot(time, comVel*50, label=' com velocity')
bax3.set_ylabel('Velocity (mm/s)')
bax3.set_xlabel('Time (s)')
bax3.set_title('COM Velocity')
bax3.axhline()

import prediction.provenance as prov
prov.stamp(bax3, .55, .15)

plt.show()

#Just compare some of the behavior
behFig3 = plt.figure(figsize=[24, 12])
behGs3 = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig3)
b3ax1 = behFig3.add_subplot(behGs3[0,0])
b3ax2 = behFig3.add_subplot(behGs3[1,0], sharex=b3ax1)
b3ax3 = behFig3.add_subplot(behGs3[2,0], sharex=b3ax1)


for neuron in AVA_like_neurons:
    b3ax1.plot(time, pos_deriv[neuron, :], label='Neuron %d' % neuron)

b3ax1.set_ylabel('Motion-corrected fluorescence intensity F')
b3ax1.set_xlabel('Time (s)')
b3ax1.legend()
b3ax1.set_title('(+) Rectified activity of AVA-Like Neurons')

b3ax2.plot(time, vel, label='eigenworm velocity')
b3ax2.set_title('Eigenworm Based Velocity')
b3ax2.axhline()

b3ax3.plot(time, comVel*50, label=' com velocity')
b3ax3.set_ylabel('Velocity (mm/s)')
b3ax3.set_xlabel('Time (s)')
b3ax3.set_title('COM Velocity')
b3ax3.axhline()

import prediction.provenance as prov
prov.stamp(b3ax3, .55, .15)

plt.show()


print("Saving plots of fav neurons to PDF...")

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(key + "_" + idn + "_selected_neurons.pdf")
numFigs = plt.gcf().number + 1
for fig in xrange(1, numFigs): ## will open an empty extra figure :(
    print("Saving Figure %d of %d" % (fig, numFigs))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved.")
