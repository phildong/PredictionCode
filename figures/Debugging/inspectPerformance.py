"""
Created Wed 8 January 2020
Gaol here is to generate useful output to understand how ro why changes to preprocessing
are affecting the performance of the different model predictions.
by Andrew Leifer
leifer@princeton.edu
"""

################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh




data = {}
for typ in ['AML32', 'AML18', 'AML175', 'AML70']:
    for condition in ['moving', 'chip']:  # ['moving', 'immobilized', 'chip']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path, '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path, 'Analysis/{}_{}.hdf5'.format(typ, condition))

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
print 'Done reading data.'


#Two goals to start wtih.
#Goal 1: is to plot perforamance of each dataset for best single neuron, PCA and SLM
#in such a way that we can do within dataset comparisons across models.
#This should be a plot of 3point line-graphs, one for each model.

import matplotlib.pyplot as plt


fig=plt.figure(1,[12, 10])
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
axes = [ax1, ax2]



b_cnt = 0
titles = ['Velocity', 'Turning']
labels = ['PCA', 'SLM', 'Best Neuron']

for behavior in ['AngleVelocity', 'Eigenworm3']:
    scores = []
    #For each type of recording
    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['analysis']
        results_from_one_dataset = []

        #For each recording
        for idn in dset.keys():

            results_pca = dset[idn]['PCAPred'][behavior]
            results_SLM = dset[idn]['ElasticNet'][behavior]
            results_from_one_dataset = np.array([results_pca['scorepredicted'], results_SLM['scorepredicted'], np.max(results_SLM['individualScore'])])

            #Plot
            axes[b_cnt].plot(np.arange(0,len(results_from_one_dataset)), results_from_one_dataset, marker=marker, label=idn)

            axes[b_cnt].title.set_text(titles[b_cnt])
            axes[b_cnt].set_ylim([0, 0.7])
            axes[b_cnt].set_xticks(np.arange(0,len(results_from_one_dataset)))
            axes[b_cnt].set_xticklabels(labels)
            axes[b_cnt].legend()
            axes[b_cnt].set_ylabel('R^2')



    b_cnt = b_cnt+ 1


import prediction.provenance as prov
prov.stamp(ax2,.55,.15)
plt.rcParams["figure.figsize"] = (20,3)

#Goal 2: show the predictions explicilty for all recordings as compared to true
# here we will probably have to generate PDFs or PPTs or somethign
# because I assume its a lot of plots





