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

#Goal 2: show the predictions explicilty for all recordings as compared to true
# here we will probably have to generate PDFs or PPTs or somethign
# because I assume its a lot of plots





# For each type of recording
fig_cnt = 1
for key, marker in zip(['AML32_moving', 'AML70_chip'], ['o', "^"]):
    dset = data[key]['input']

    # For each recording
    for idn in dset.keys():
        fig_cnt=fig_cnt+1
        fig = plt.figure(fig_cnt, [28, 12])
        fig.suptitle([key + ' ' + idn])
        row = 3
        col = 2
        axes = []
        for each in np.arange(row * col)+1:
            axes = np.append(axes, plt.subplot(row, col, each))


        ax_cnt =-1
        for flag, pred_type in zip(['PCAPred', 'ElasticNet', 'ElasticNet'], ['PCA', 'SLM', 'Best Neuron']):
            for behavior, title in zip(['AngleVelocity', 'Eigenworm3'],  ['Velocity', 'Turn']):
                ax_cnt = ax_cnt + 1

                #Get the data
                moving = data[key]['input'][idn]
                valid_map = moving['Neurons']['I_valid_map']
                movingAnalysis = data[key]['analysis'][idn]

                splits = movingAnalysis['Training']

                indices_contig_trace_test = valid_map[splits[behavior]['Test']]
                train, test = splits[behavior]['Train'], splits[behavior]['Test']

                beh = moving['BehaviorFull'][behavior]

                time = moving['Neurons']['I_Time']
                time_crop_noncontig = moving['Neurons']['I_Time_crop_noncontig']

                behPred = np.empty(moving['BehaviorFull'][behavior].shape)
                behPred[:] = np.nan
                additional_title_text = ''

                if pred_type == 'Best Neuron':
                    #Find the best Neuron
                    nid = np.argmax(movingAnalysis[flag][behavior]['individualScore'])
                    bestNeur= moving['Neurons']['I_smooth_interp_crop_noncontig'][nid,:]

                    bestNeur_scaled = (bestNeur - np.nanmean(bestNeur))  * np.nanstd(beh) / np.nanstd(bestNeur) + np.nanmean(beh)

                    behPred[valid_map] = bestNeur_scaled

                    additional_title_text = ", Neuron ID: %i" % nid

                else:
                    behPred[valid_map] = movingAnalysis[flag][behavior]['output']


                #Actually Plot
                axes[ax_cnt].plot(time, beh, label="Measured")
                axes[ax_cnt].plot(time, behPred, label="Predicted")

                axes[ax_cnt].title.set_text([pred_type + ', ' + title + additional_title_text])
                axes[ax_cnt].legend()
                axes[ax_cnt].set_xlabel('Time (s)')
                axes[ax_cnt].set_xlim( [time[valid_map[0]], time[valid_map[-1]]])

                axes[ax_cnt].axvspan(time_crop_noncontig[test[0]], time_crop_noncontig[test[-1]], color='gray', zorder=-10,
                            alpha=0.1)
                #ax7.axhline(color='k', linestyle='--', zorder=-1)



                #axscheme2.text(t[-1], yl + yo, \
                 #              r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])),
                 #              horizontalalignment='right')
        prov.stamp(axes[ax_cnt],.55,.15)


print("Saving figures to pdf...")

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved.")