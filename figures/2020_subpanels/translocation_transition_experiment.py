
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
#Goal is to plot neural trajectories projected into first three PCs


def plot_trajectories(pcs, title='Neural State Space Trajectories', color='blue'):
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    for nplot in np.arange(6) + 1:
        ax = plt.subplot(2, 3, nplot, projection='3d')
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2])
        ax.view_init(np.random.randint(360), np.random.randint(360))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    import prediction.provenance as prov
    prov.stamp(plt.gca(), .55, .15, __file__)
    return


for typ_cond in ['AKS297.51_transition']: #, 'AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # Hard coding in folder for old dataset:
    #folder = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/'
    #dataLog = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/Special_transition_datasets.txt'

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
    theDataset = '193044'
    transition = im_start = 950
    im_end = 2885


    for key in filter(lambda x: theDataset in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']

        X = dataSets[key]['BehaviorFull']['X']  # Get the X position
        Y = dataSets[key]['BehaviorFull']['Y']  # Get they Y position
        X = X - X[0]  # center the beginning of each recording at the origin
        Y = Y - Y[0]  # center the beginning of each recording at the origin
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ### Plot the lines to that they chang ecolor
        points = np.array([X, Y]).transpose().reshape(-1, 1, 2)
        # set up a list of segments
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        # see what we've done here -- we've mapped our (x,y)
        # points to an array of segment start/end coordinates.
        # segs[i,0,:] == segs[i-1,1,:]
        # make the collection of segments
        from matplotlib.collections import LineCollection

        lc = LineCollection(segs, cmap=plt.get_cmap('gist_rainbow'), linewidths=4)
        lc.set_array(time)  # color the segments by our parameter
        # plot the collection
        ax.add_collection(lc)  # add the collection to the plot
        ax.plot([0, 0], [0, 1], 'k')

        #    ax.plot(X_data[key][ind], Y_data[key][ind], label="furthest_distance", color="orange")
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)

        import prediction.provenance as prov

        prov.stamp(ax, .55, .15, __file__)
        fig.colorbar(lc)  # , cax=ax)  # , orientation='horizontal')
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)

print('yo')
