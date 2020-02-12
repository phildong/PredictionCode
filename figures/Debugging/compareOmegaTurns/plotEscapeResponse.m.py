print("Loading timing information of escape responses from CSV file ...")

import numpy as np

my_dtype=['S512', 'S512', 'f8', 'int32', 'int32', 'int32', 'int32', 'S12', 'int32', 'int32', 'int32', 'int32', 'int32']

timing_info = np.genfromtxt('realtiveTimingWithVolumes.txt',
                        dtype=my_dtype, names=True, delimiter=',')
print("Loaded!")


#We need to load the data.
# But of course the CSV spreadsheet with timing info generated from MATLAB on tigress has different paths
# then the local analysis files. So we will read in everything and then do some kindof matching.



#Loading the data the usual way. No immobile needed

import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh

#Load recorded datasets
print("Load Rrecorded datasets.")
data = {}
for typ in ['AML32',  'AML70']:
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
print('Done reading data.')




import matplotlib.pyplot as plt

sumfig, sumax = plt.subplots(1,3, figsize=[18, 5])
sumfig.suptitle('All escape responses (including mulitple per recording)')


#Loop through the escape response instances identifed in the CSV file
for each in np.arange(timing_info.shape[0]):

    #Loop through all  classes of recordings
    for key in ['AML32_moving', 'AML70_chip', 'AML70_moving']:
        #Loop through all recordings
        dset = data[key]['analysis']
        for idn in dset.keys():
            #If the recording is the the one that belongs to the current escape response timing info
            if idn in timing_info['BrainScannerFolder'][each]: #Compare string of BrainScanner folder with path to folder from Tigress
                #Go ahead and do stuff.
                print(idn + " is in " + timing_info['BrainScannerFolder'][each])

                #Setup figure
                fig, ax = plt.subplots(1, 3, figsize=[18, 5])
                fig.suptitle(key + ' ' + idn +
                               ' ' + np.str(timing_info['StartVolume'][each])
                              + ' to ' +
                              np.str(timing_info['EndVolume'][each]))

                ##############
                #Plot Behavior
                #############
                vel = data[key]['input'][idn]['BehaviorFull']['AngleVelocity']
                curve = data[key]['input'][idn]['BehaviorFull']['Eigenworm3']
                t = np.arange(timing_info['StartVolume'][each],
                                timing_info['EndVolume'][each] + 1)
                #ax[0].plot(curve[t], vel[t])

                #Plot lines that change color with time from a Stackoverflow Question
                from matplotlib.collections import LineCollection

                points = np.array([curve[t], vel[t]]).transpose().reshape(-1, 1, 2)

                # set up a list of segments
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                # see what we've done here -- we've mapped our (x,y)
                # points to an array of segment start/end coordinates.
                # segs[i,0,:] == segs[i-1,1,:]

                # make the collection of segments
                lc = LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'))
                lc.set_array(t)  # color the segments by our parameter

                # plot the collection
                ax[0].add_collection(lc)  # add the collection to the plot


                #Add plot formatting
                vel_lims = [-2.5, 4]
                curve_lims = [-25, 25]
                ax[0].set_ylim(vel_lims)
                ax[0].set_title('Measured Behavior')
                ax[0].set_xlim(curve_lims)
                ax[0].set_xlabel('Curvature')
                ax[0].set_ylabel('Velocity')
                ax[0].axhline(linewidth=0.5, color='k')
                ax[0].axvline(linewidth=0.5, color='k')


                # plot the collection
                ax[0].add_collection(lc)  # add the collection to the plot

                ax[0].plot(curve[t[0]],vel[t[0]],
                           marker='>', fillstyle='full',
                           color='red')
                ax[0].plot(curve[t[-1]], vel[t[-1]],
                           marker='s', fillstyle='full',
                           color='#ff3399')


                #Also plot the behavior in the figure that aggregates all the esecape responses and recordings
                lc_alpha =LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'), alpha=0.3)
                lc_alpha.set_array(t)  # color the segments by our parameter

                sumax[0].add_collection(lc_alpha)  # add the collection to the plot



                sumax[0].set_ylim(vel_lims)
                sumax[0].set_title('Measured Behavior')
                sumax[0].set_xlim(curve_lims)
                sumax[0].set_xlabel('Curvature')
                sumax[0].set_ylabel('Velocity')
                sumax[0].axhline(linewidth=0.5, color='k')
                sumax[0].axvline(linewidth=0.5, color='k')

                ###########
                #Plot PCA
                ############

                import matplotlib.pyplot as plt
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                # Plot neural state space trajectories in first 3 PCs
                # also reduce dimensionality of the neural dynamics.
                nComp = 3  # pars['nCompPCA']
                pca = PCA(n_components=nComp)
                Neuro = np.copy(data[key]['input'][idn]['Neurons']['I_smooth_interp_crop_noncontig']).T


                # make sure data is centered
                sclar = StandardScaler(copy=True, with_mean=True, with_std=False)
                Neuro_mean_sub = sclar.fit_transform(Neuro)

                pcs = pca.fit_transform(Neuro_mean_sub)

                #We need some way from converting our volume number indices to indices
                # of the I_smooth_interp_crop_noncontig...
                valid_map = data[key]['input'][idn]['Neurons']['I_valid_map']

                def find_nearest_idx(array, value):
                    array = np.asarray(array)
                    idx = (np.abs(array - value)).argmin()
                    return idx

                #indices
                t_noncont = np.arange(find_nearest_idx(valid_map, t[0]),
                                        find_nearest_idx(valid_map, t[-1]) + 1)

                #actual time points
                time_crop_noncontig = data[key]['input'][idn]['Neurons']['I_Time_crop_noncontig']

                #Plot PCA trajectories of first two PC modes
                #ax[1].plot(pcs[t_noncont, 0], pcs[t_noncont, 1])



                #Plot lines that change color with time from a Stackoverflow Question
                from matplotlib.collections import LineCollection

                #Load in hand annotated rotation information to make the PC plots best match behavior
                if timing_info['PC_SwapAxes'][each] == 0:
                    x_pc = 0 #PC1 goes horizontal
                    y_pc = 1 #PC2 goes vertical
                else:
                    x_pc = 1 #PC2 goes horizontal
                    y_pc = 0 #PC1 goes veritcal
                points = np.array([pcs[t_noncont, x_pc], pcs[t_noncont, y_pc]]).transpose().reshape(-1, 1, 2)

                # set up a list of segments
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                # see what we've done here -- we've mapped our (x,y)
                # points to an array of segment start/end coordinates.
                # segs[i,0,:] == segs[i-1,1,:]

                # make the collection of segments
                lc = LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'))
                lc.set_array(time_crop_noncontig[t_noncont])  # color the segments by our parameter

                # plot the collection
                ax[1].add_collection(lc)  # add the collection to the plot

                #Add plot formatting

                ax[1].set_title('Unoptimized PCs (PCA)')
                ax[1].set_xlabel('PC%d' % (x_pc+1))
                ax[1].set_ylabel('PC%d' % (y_pc+1))
                ax[1].axhline(linewidth=0.5, color='k')
                ax[1].axvline(linewidth=0.5, color='k')


                #Load in hand annotated reflection information to make the PC plots best match behavior
                if timing_info['PC_Flip_NewX'][each] == 1:
                    flipx=1
                else:
                    flipx=0

                if timing_info['PC_Flip_NewY'][each] == 1:
                    flipy=1
                else:
                    flipy=0


                ax[1].set_xlim(np.roll([np.min(pcs[t_noncont, x_pc]), np.max(pcs[t_noncont, x_pc])], flipx))
                ax[1].set_ylim(np.roll([np.min(pcs[t_noncont, y_pc]), np.max(pcs[t_noncont, y_pc])], flipy))


                ########################
                #Plot SLM trajectories
                #########################
                flag = 'ElasticNet'
                velPred = data[key]['analysis'][idn][flag]['AngleVelocity']['output']
                curvPred = data[key]['analysis'][idn][flag]['Eigenworm3']['output']

                # Plot lines that change color with time from a Stackoverflow Question
                from matplotlib.collections import LineCollection

                points = np.array([curvPred[t_noncont], velPred[t_noncont]]).transpose().reshape(-1, 1, 2)

                # set up a list of segments
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                # see what we've done here -- we've mapped our (x,y)
                # points to an array of segment start/end coordinates.
                # segs[i,0,:] == segs[i-1,1,:]

                # make the collection of segments
                lc = LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'))
                lc.set_array(time_crop_noncontig[t_noncont])  # color the segments by our parameter

                # plot the collection
                ax[2].add_collection(lc)  # add the collection to the plot

                #plot start and end point
                ax[2].plot(curvPred[t_noncont[0]],velPred[t_noncont[0]],
                           marker='>', fillstyle='full',
                           color='red')
                ax[2].plot(curvPred[t_noncont[-1]], velPred[t_noncont[-1]],
                           marker='s', fillstyle='full',
                           color='#ff3399')

                # Add plot formatting

                ax[2].set_title('SLM \n(interp over all NaNs)')
                ax[2].set_xlabel('Predicted Curvature')
                ax[2].set_ylabel('Predicted Velocity')
                ax[2].axhline(linewidth=0.5, color='k')
                ax[2].axvline(linewidth=0.5, color='k')
                ax[2].set_xlim(curve_lims)
                ax[2].set_ylim(vel_lims)


                #NOTE WE ARE INTERPOLATING OVER NANS HERE

                #Also plot the SLM predictions in the figure that aggregates all the esecape responses and recordings
                lc_alpha =LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'), alpha=0.3)
                lc_alpha.set_array(t)  # color the segments by our parameter

                sumax[2].add_collection(lc_alpha)  # add the collection to the plot

                sumax[2].set_title('SLM \n(interp over all NaNs)')
                sumax[2].set_xlabel('Predicted Curvature')
                sumax[2].set_ylabel('Predicted Velocity')
                sumax[2].axhline(linewidth=0.5, color='k')
                sumax[2].axvline(linewidth=0.5, color='k')
                sumax[2].set_xlim(curve_lims)
                sumax[2].set_ylim(vel_lims)



                import prediction.provenance as prov
                prov.stamp(ax[2], .55, .15)



prov.stamp(sumax[1], .55, .15)





print("Generating PDF of escape response plots")
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("escape_response.pdf")
for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved escape response plots.")
