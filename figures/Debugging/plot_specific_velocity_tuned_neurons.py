
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
activity = dset['Neurons']['I_smooth']
time = dset['Neurons']['I_Time']
gRaw = dset['Neurons']['gRaw']
numNeurons = activity.shape[0]
vel = dset['BehaviorFull']['AngleVelocity']
comVel = dset['BehaviorFull']['CMSVelocity']
curv = dset['BehaviorFull']['Eigenworm3']




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




#Just compare some of the behavior
behFig2 = plt.figure(figsize=[24, 12])
behGs2 = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig2
                           )
b2ax1 = behFig2.add_subplot(behGs2[0,0])
b2ax2 = behFig2.add_subplot(behGs2[1,0], sharex=b2ax1)
b2ax3 = behFig2.add_subplot(behGs2[2,0], sharex=b2ax1)

b2ax1.plot(activity[11, :], label='Neuron 11')
b2ax1.plot(activity[24, :], label='Neuron 24')
b2ax1.plot(activity[74, :], label='Neuron 85')
b2ax1.plot(activity[30, :], label='Neuron 30')
b2ax1.set_ylabel('Motion-corrected fluorescence intensity F')
b2ax1.set_xlabel('Volume (6 vol/s)')
b2ax1.legend()

b2ax1.set_title('Activity of Select Neurons')

b2ax2.plot(vel, label='eigenworm velocity')
b2ax2.set_title('Eigenworm Based Velocity')
b2ax2.axhline()

b2ax3.plot(comVel*50, label=' com velocity')
b2ax3.set_title('COM Velocity')
b2ax3.set_ylabel('Velocity (mm/s)')
b2ax3.axhline()

import prediction.provenance as prov
prov.stamp(b2ax3, .55, .15)


plt.show()



plt.figure(figsize=[10,10])
plt.plot(vel, comVel*50)
plt.show()


#Just compare some of the behavior
behFig = plt.figure(figsize=[24, 12])
behGs = gridspec.GridSpec(ncols=1, nrows=3, figure=behFig)
bax1 = behFig.add_subplot(behGs[0,0])
bax2 = behFig.add_subplot(behGs[1,0], sharex=bax1)
bax3 = behFig.add_subplot(behGs[2,0], sharex=bax1)

bax1.plot(activity[53, :],label='Neuron 53 (AVA)')
bax1.plot(activity[56, :],label='Neuron 56')
bax1.plot(activity[59, :],label='Neuron 59')
bax1.plot(activity[1, :],label='Neuron 1')
bax1.plot(activity[33, :],label='Neuron 33')
bax1.plot(activity[49, :],label='Neuron 49')
bax1.set_ylabel('Motion-corrected fluorescence intensity F')
bax1.set_xlabel('Volume (6 vol/s)')
bax1.legend()
bax1.set_title('Activity of Select Neurons')

bax2.plot(vel, label='eigenworm velocity')
bax2.set_title('Eigenworm Based Velocity')
bax2.axhline()

bax3.plot(comVel*50, label=' com velocity')
bax3.set_ylabel('Velocity (mm/s)')
bax3.set_xlabel('Volumes (6 vol/s)')
bax3.set_title('COM Velocity')
bax3.axhline()

import prediction.provenance as prov
prov.stamp(bax3, .55, .15)

plt.show()


