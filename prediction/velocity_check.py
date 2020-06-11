import SLM
import MARS
from Classifier import rectified_derivative
import userTracker
import dataHandler as dh

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.backends.backend_pdf

import numpy as np
from sklearn.decomposition import PCA

import os

excludeSets = {'BrainScanner20200309_154704'}
excludeInterval = {''}

pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "velocity_check.pdf"))
for typ_cond in ['AML32_moving', 'AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # data parameters
    dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
            'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5,  # gauss window for red and green channel
            'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
            'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
            }
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
    keyList = np.sort(dataSets.keys())

    for key in keyList:
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        cmsvelocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']

        fig = plt.figure(constrained_layout = True, figsize=(10,20))
        gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=(1, 1))

        adj = lambda x: (x-np.mean(x))/np.std(x)
        sc = fig.add_subplot(gs[:2,:])
        avel = adj(velocity)
        acms = adj(cmsvelocity)
        sc.plot(avel, acms, 'bo')
        sc.plot([min(avel), max(avel)], [min(acms), max(acms)], 'k-.')
        sc.set_xlabel('Centerline Velocity', fontsize=14)
        sc.set_ylabel('CMS Velocity', fontsize=14)

        vel = fig.add_subplot(gs[2, :])
        vel.plot(time, velocity, 'k', lw=1)
        vel.set_xlabel('Time', fontsize=14)
        vel.set_ylabel('Centerline Velocity', fontsize=14)

        cms = fig.add_subplot(gs[3, :])
        cms.plot(time, cmsvelocity, 'k', lw=1)
        cms.set_xlabel('Time', fontsize=14)
        cms.set_ylabel('CMS Velocity', fontsize=14)

        rho2 = np.corrcoef(avel, acms)[0,1]**2

        fig.suptitle(r"%s ($\rho^2 = %0.2f$)" % (key, rho2))

        pdf.savefig(fig)

pdf.close()