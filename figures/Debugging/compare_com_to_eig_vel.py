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
velnc = dset['Behavior_crop_noncontig']['AngleVelocity']
comVelnc = dset['Behavior_crop_noncontig']['CMSVelocity']
curv = dset['Behavior_crop_noncontig']['Eigenworm3']
vel = dset['BehaviorFull']['AngleVelocity']
comVel = dset['BehaviorFull']['CMSVelocity']
X = dset['BehaviorFull']['X']  # Get the X position
Y = dset['BehaviorFull']['Y']  # Get they Y position

crude_vel=np.sqrt(np.square(np.diff(np.squeeze(X)))+np.square(np.diff(np.squeeze(Y))))

import matplotlib.pyplot as plt

Factor = 200 #The CMSvelocity seems to be off of mm/s by a factor of 200
from scipy.ndimage import gaussian_filter1d
#muptiply by 6 to get in mm/ s, gaussian smooth by three
mycomVel = gaussian_filter1d(crude_vel*6,3)
comVel = comVel* Factor

plt.figure()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.plot(vel, comVel, 'o', color='black', alpha=0.1)
plt.xlabel('Eigen Velocity')
plt.ylabel('Center of Mass Velocity')

m, b = np.polyfit(vel, comVel, 1)
plt.plot(vel, m*vel + b, 'r--')
plt.show()


print("done")