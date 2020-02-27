
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
numNeurons = activity.shape[0]
vel = dset['BehaviorFull']['AngleVelocity']
curv = dset['BehaviorFull']['Eigenworm3']



import matplotlib.pyplot as plt

#Loop through each neuron
for neuron in np.random.permutation(np.arange(numNeurons)):

    fig, ax = plt.subplots(2, 2, figsize=[10, 10])
    ax= ax.flatten()

    fig.suptitle(key + ' ' + idn + ' Neuron: #' + str(neuron), color='w')

    #Randomize the axes order withour replacement
    randAx = np.random.permutation(np.arange(2))
    #Actually Plot
    ax[randAx[0]].plot(vel, activity[neuron, :],'o', markersize=0.7)
    ax[randAx[1]].plot(vel, np.roll(activity[neuron, :],
                                     np.random.randint(0, activity.shape[1])),
                       'o', markersize=0.7)
    ax[randAx[0]].set_title('TRUE', color='w')
    ax[randAx[1]].set_title('SHUFFLE', color='w')

    # Randomize the axes order withour replacement
    randAx = np.random.permutation(np.arange(2,4))
    ax[randAx[0]].plot(curv, activity[neuron, :],'o', markersize=0.7)
    ax[randAx[1]].plot(curv, np.roll(activity[neuron, :],
                        np.random.randint(0, activity.shape[1])),
                       'o', markersize=0.7)
    ax[randAx[0]].set_title('TRUE', color='w')
    ax[randAx[1]].set_title('SHUFFLE', color='w')


    #Add line for behavior = 0
    ax[0].axvline(linewidth=0.5, color='k')
    ax[1].axvline(linewidth=0.5, color='k')
    ax[2].axvline(linewidth=0.5, color='k')
    ax[3].axvline(linewidth=0.5, color='k')

    ax[0].set_xlabel('Velocity')
    ax[1].set_xlabel('Velocity')
    ax[2].set_xlabel('Curvature')
    ax[3].set_xlabel('Curvature')



    ax[0].set_ylabel('Fluorescence (common-noise rejected)')
    ax[1].set_ylabel('Fluorescence (common-noise rejected)')
    ax[2].set_ylabel('Fluorescence (common-noise rejected)')
    ax[3].set_ylabel('Fluorescence (common-noise rejected)')



#Later should sort by COM of the distribution.. but for now just plot


print("Saving tuning curve plots to PDF...")

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(key + "_" + idn + "_tuning.pdf")
numFigs = plt.gcf().number + 1
for fig in xrange(1, numFigs): ## will open an empty extra figure :(
    print("Saving Figure %d of %d" % (fig, numFigs))
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved.")








