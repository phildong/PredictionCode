# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:10 2017
load and compare multiple data sets for behavior prediction.
@author: monika scholz
"""
from __future__ import division # make division give you floats, as is standard for python3
import scipy.io
import os
import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
#from scipy.signal import medfilt, deconvolve
from skimage.transform import resize
from sklearn import preprocessing
import makePlots as mp
from scipy.ndimage.filters import gaussian_filter1d
import h5py
from scipy.special import erf
#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,  FastICA



def recrWorm(av, turns, thetaTrue, r, show = 0):
    """recalculate eigenworm prefactor from angular velocity and turns."""
    thetaCum = np.cumsum(np.copy(av)/6.) # division due to rate: it's velocity per secod instread of per volume
    # reset theta every minute to real value
    dt = np.arange(50, len(thetaCum), 60)
    for tt in dt:    
        thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
    
    radius = np.zeros(len(av)) 
    for tt in dt:
        radius[tt:] = r[tt]
#    plt.plot(radius)
#    plt.show()
    r = radius
#    thetaCum -= thetaCum[50]-thetaTrue[50] 
#    thetaCum -= np.mean(thetaCum)- np.mean(thetaTrue)
    #tt = 0
    #thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
    # recalculate the phase angle from cumulative phase
    theta = np.mod(thetaCum, 2*np.pi)#-np.pi
    sign = np.ones(len(theta))
    sign[np.where(np.abs(theta-np.pi)>np.pi/2.)] = -1
    # do the same for the true angle
    thetaD = np.mod(thetaTrue, 2*np.pi)
    thetaDS = np.where(np.abs(thetaD-np.pi)>np.pi/2., -thetaD, thetaD)
    if show:
        plt.figure('Real and reconstructed phase angle')
        plt.subplot(221)
        plt.plot(thetaCum, label = 'reconstructed')
        plt.plot(thetaTrue, label = 'real')
        
        #plt.scatter(thetaCum,thetaTrue, label = 'reconstructed')
        #plt.plot(thetaTrue, label = 'real')
        plt.ylabel('Accumulated phase angle')
        plt.legend()
        plt.subplot(222)
        plt.plot(thetaCum-thetaTrue, label = 'residuals')
        plt.plot(np.cumsum(thetaCum-thetaTrue), label = 'cumulative residuals')
        plt.ylabel('Phase difference')
        plt.legend()
        plt.subplot(223)
        plt.plot(theta, label = 'reconstructed')
        plt.plot(thetaD, label ='real')#-np.pi)
        plt.ylabel('Phase angle (rad)')
        plt.xlabel('Time (Volumes)')
        plt.legend()
        plt.subplot(224)
        plt.scatter(sign*theta,thetaDS,s=1, alpha=0.1)
        plt.ylabel('Phase angle (rad)')
        plt.xlabel('reconstructed Phase angle (rad)')
        plt.tight_layout()
        plt.show(block=True)
    # recalculate eigenworms
    x = -np.sqrt(r)*np.tan(sign*theta)/np.sqrt((np.tan(sign*theta)**2+1))
    y = -sign*np.sqrt(r-x**2)
    return x,y, turns



def makeEthogram(anglevelocity, pc3):
    """use rotated Eigenworms to create a new Ethogram."""
    etho = np.zeros((len(anglevelocity),1))
    # set forward and backward
    etho[np.where(anglevelocity>0.05)] = 1
    etho[np.where(anglevelocity<=-0.05)] = -1
    # overwrite this in case of turns
    etho[np.abs(pc3)>10] = 2
    return etho

def loadPoints(folder,  straight = True):
    """get tracked points from Pointfile."""
    points = np.squeeze(scipy.io.loadmat(os.path.join(folder,'pointStatsNew.mat'))['pointStatsNew'])
    
    if straight:
        return [p[1] for p in points]
    else:
        return [p[2] for p in points]
    
def loadEigenBasis(filename, nComp=3, new = True):
    """load the specific worm basis set."""
    if new:
        eigenworms = np.loadtxt(filename)[:nComp]
    else:
        eigenworms = scipy.io.loadmat(filename)['eigbasis'][:nComp]
    # ncomponents controls how many we use
    eigenworms = resize(eigenworms, (nComp,99))
    return eigenworms



def calculateEigenwormsFromCL(cl, eigenworms):
    """takes (x,y) pairs from centerlines and returns eigenworm coefficients."""
    # coordinate segments
    diffVec = np.diff(cl, axis=1)
    # calculate tangential vectors
    wcNew = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
    #################these things are needed for reconstruction
    # get mean angle
    meanAngle = np.mean(wcNew, axis=1)
    # get segment lengths
    lengths = np.sqrt(diffVec[:,:,1]**2+diffVec[:,:,0]**2)
    # get overall alignment in space
    # reference point to start with
    refPoint = cl[:,0]
    # calculate mean subtracted tangent angles
    wcNew = wcNew-meanAngle[:, np.newaxis]
    # project onto Eigenworms
    pcsNew = np.dot(eigenworms,wcNew.T)
    return pcsNew, meanAngle, lengths, refPoint

def calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint):
    """takes eigenworms and a few reference parameters to recreate centerline."""
    # now we recreate the worm
    wcR = np.dot(pcsNew.T, eigenworms) + meanAngle[:, np.newaxis] 
    # recreate tangent vectors with correct length
    tVecs = np.stack([lengths*np.cos(wcR), -lengths*np.sin(wcR)], axis=2)
    # start at same point as original CL
    clApprox = refPoint[:, np.newaxis] + np.cumsum(tVecs, axis=1)
    return clApprox       

def loadCenterlines(folder, full = False, wormcentered = False):
    """get centerlines from centerline.mat file"""
    #cl = scipy.io.loadmat(folder+'centerline.mat')['centerline']
    tmp =scipy.io.loadmat(os.path.join(folder,'centerline.mat'))
    
    cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['centerline'], 2,0)
    if wormcentered:
        cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['wormcentered'], 1,0)

    tmp = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    
    clTime = np.squeeze(tmp['clTime']) # 50Hz centerline times
    volTime =  np.squeeze(tmp['hasPointsTime'])# 6 vol/sec neuron times
    
    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime))))  
    if not full:
        # reduce to volume time
        cl = cl[clIndices.astype(int)]
    #wcNew = wc[clIndices.astype(int)]
    #epNew = ep[clIndices.astype(int)]
    
#    for cl in clNew[::10]:
#        plt.plot(cl[:,0], cl[:,1])
#    plt.show()
    return cl ,clIndices.astype(int), clTime
    
def transformEigenworms(pcs, dataPars):
    """interpolate, smooth Eigenworms and calculate associated metrics like velocity."""
    pc3, pc2, pc1 = pcs
    #mask nans in eigenworms by linear interpolation
    for pcindex, pc in enumerate(pcs):
        # mask nans by linearly interpolating
        mask = np.isnan(pc1)

        if np.any(mask):
            pc[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pc1[~mask])
            import warnings
            warnings.warn('Found ' + str(np.sum(mask)) + 'NaNs that are being interpolated over in the behavior PC1, PC2 or PC3.')
        pcs[pcindex] = pc
        
    theta = np.unwrap(np.arctan2(pcs[2], pcs[1]))
    # convolution with gaussian kernel derivative
    velo = gaussian_filter1d(theta, dataPars['gaussWindow'], order=1)
    # velo is in radians/frame

    accel = gaussian_filter1d(theta, 2*dataPars['gaussWindow'], order=2)

#Andy commented this out Jan 2020
 #   for pcindex, pc in enumerate(pcs):
 #       pcs[pcindex] = gaussian_filter1d(pc, dataPars['medianWindow'])
    return pcs, velo, theta, accel

def decorrelateNeuronsLinear(R, G):
        # Xinwei Yu's linear motion correction algorithm
        # Re-implemented by Andrew Leifer, based on an implimentation by Matthew Creamer
        I = np.empty(G.shape)
        I[:] = np.nan  # initialize it with nans

        # the fits can't handle the nans, so we gotta tempororaily exclude them
        # we won't interpolate, we will just remove them than add them back
        nanmask = np.logical_or(np.isnan(R), np.isnan(G))

        for n in range(R.shape[0]): #for each neuron
            red_column = np.expand_dims(R[n, ~nanmask[n]].T, axis=1)
            red_with_ones = np.concatenate([red_column, np.ones(red_column.shape)], axis=1)
            best_fit, _, _, _ = np.linalg.lstsq(red_with_ones, G[n, ~nanmask[n]].T, rcond=None)
            I[n, :] = G[n, :] - (best_fit[0] * R[n, :] + best_fit[1])

            if False:
                plt.figure()
                plt.plot(I[n,:], label='I')
                plt.plot(G[n,:]-np.nanmean(G[n,:]), label='G')
                plt.legend()
        return I







def loadData(folder, dataPars, ew=1, cutVolume = None):
    """load matlab data."""
    print 'Loading ', folder
    try:
        data = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    except IOError:
        print 'IOERROR'
        data = scipy.io.loadmat(os.path.join(folder,'heatData.mat'))
    # unpack behavior variables
    ethoOrig, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T
    # Jan 2020: andy suspects this "vel" variable comes from the COM of the centerline +
    # the stage position, as Jeff calculates (and smooths) here:
    # https://github.com/leiferlab/3dbrain/blob/e38908c6dd5a4ae2829946ab91b9a00af9516f2e/fiducialCropper3.m#L57
    #
    # for the contents of the MATLAB behavior struct, see https://github.com/leiferlab/3dbrain/blob/9a5a8d6b071c1c9e49a86e66e2064e9ebe69224d/rerunDataCollectionMS.m#L142

    if cutVolume is None:
        cutVolume = np.max(vel.size)

    # get centerlines with full temporal resolution of 50Hz
    clFull, clIndices, clTimes = loadCenterlines(folder, full=True)
    curv = getCurvature(clFull)


    # load new eigenworms
    import userTracker
    codePath = userTracker.codePath()
    eigenworms = loadEigenBasis(filename = os.path.join(codePath,'utility/Eigenworms.dat'), nComp=3, new=True)
    # get full set of Eigenworms
    pcsFull, meanAngle, lengths, refPoint = calculateEigenwormsFromCL(clFull, eigenworms)
    # do Eigenworm transformations and calculate velocity etc. 
    pcs, velo, theta, accel = transformEigenworms(pcsFull, dataPars)

    #calculate phase velocity, this is the velocity of the body bends as tehy propogate along the worm's body in units of bodylengths / s
    vel_ps = findPhaseVelocity(curv, clTimes, sigma=50)
    curv_metric = get_curv_metric(curv, ROI_start=15, ROI_end=80, num_stds=6)

    debug_findPhaseShift(vel_ps, velo, vel, clIndices, clTimes, curv, folder=folder)
    #debug_curvature_metrics(curv, pcs[0,:])

    #downsample to 6 volumes/sec
    pc3, pc2, pc1 = pcs[:,clIndices]
    vel_ps_downsampled = vel_ps[clIndices]
    curv_metric_dowsampled = curv_metric[clIndices]

    velo = velo[clIndices]*50. # to get it in per Volume units -> This is radians per sec
    accel = accel[clIndices]*50

    theta = theta[clIndices] # get it in per seconds
    cl = clFull[clIndices]
    # ethogram redone
    etho = makeEthogram(velo, pc3)
    #etho = np.squeeze(ethoOrig)
    # mask nans in ethogram
    ethomask = np.isnan(etho)
    if np.any(ethomask):
        etho[ethomask] = 0

    rRaw=np.array(data['rRaw'])[:,:len(np.array(data['hasPointsTime']))]
    gRaw=np.array(data['gRaw'])[:,:len(np.array(data['hasPointsTime']))]


    if cutVolume is None:
        cutVolume = rRaw.shape[1]

    idx = np.arange(rRaw.shape[1])
    idx_data = idx[idx <= cutVolume]
    idx_identities = idx[idx > cutVolume]


#    assert np.true_divide(np.sum(np.isnan(rRaw)), rRaw.shape[0]*rRaw.shape[1]) < .3, ["the Red channel is over 1/3rd NaNs. This dataset should be removed from consideration: " + folder ]
    rphotocorr = np.array(data['rPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]
    gphotocorr = np.array(data['gPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]

    #load neural data
    debug = False
    original = False
    if original:
        R = np.copy(rphotocorr)[:, idx_data]
        G = np.copy(gphotocorr)[:, idx_data]
    else:
        R = rRaw.copy()[:,  idx_data]
        G = gRaw.copy()[:, idx_data]

        vps = dataPars['volumeAcquisitionRate']

        R = correctPhotobleaching(R, vps, error_bad_fit=True)
        G = correctPhotobleaching(G, vps)

        #Choose whether we want to apply our outlier detection
        #Or Jeff's
        jeffs_outlier_detection = True
        #Note that the paper as posted on the arxiv achieves
        # very high prediction R^2 using jeff's outlier detection

        if jeffs_outlier_detection:
             # Below we apply Jeff's outlier nan'ing
             R[np.isnan(rphotocorr[:, idx_data])] = np.nan
             G[np.isnan(gphotocorr[:, idx_data])] = np.nan
             # Note: Jeff's outlier detection is pretty convoluted
             # It atually calculates the Delta R/ R0 signal
             # And then looks for outliers in that and then NaN's them out
             # And propogates them back into the rphotocorr
             # signal.
             #
             # In other words, its pretty hard to reimpliment exactly
             # Especially since we don't want to use the ratio approach
        else:
            # then we apply ours based on a subset of the criteria
            # That jeff uses
            R = nanOutliers(R, zscoreBounds=[-2,5], dataLowerBound=40, max_nan_per_col=0.5, vps=vps)
            G = nanOutliers(G, zscoreBounds=[-5,5], dataLowerBound=1, max_nan_per_col=0.5, vps=vps)

        #Remove isolated non-NaNs in a sea of NaNs
        R = close_nan_holes(R)
        G = close_nan_holes(G)


        if debug:
            nplots=2
            chosen = np.round(np.random.rand(nplots)*R.shape[0]).astype(int)
            import matplotlib.pyplot as plt
            plt.cla()
            plt.subplot(nplots, 1, 1,)
            plt.plot(G[chosen[0],:],'g', label='gRaw w/ python photocorrection')
            plt.plot(gphotocorr[chosen[0],:],'b', label='gPhotoCorr via matlab')
            plt.title('Comparsion of Green Channel photocorrection methods for neuron ' + str(chosen[0]))
            plt.legend()

            plt.subplot(nplots,1,2)
            plt.plot(R[chosen[0], :], 'r', label='rRaw w/ python photocorrection')
            plt.plot(rphotocorr[chosen[0], :], 'm', label='rPhotoCorr via matlab')
            plt.title('Comparsion of Red Channel photocorrection methods for neuron ' + str(chosen[0]))
            plt.legend()
            plt.show()

    #NaN-out Flagged volumes (presumably these are manually flagged somehwere in the pipeline)
    if 'flagged_volumes' in data.keys():
        if len(data['flagged_volumes'])>0:
            R[:, np.array(data['flagged_volumes'][0])] = np.nan
            G[:, np.array(data['flagged_volumes'][0])] = np.nan



    #Reject noise common to both Red and Green Channels using ICA
    #I = decorrelateNeuronsICA(R, G)
    I = decorrelateNeuronsLinear(R, G)
    print("After motion correction",np.nanmean(I))

    #Apply Gaussian Smoothing (and interpolate for free) #Keep the raw green around so we can use it to set heatmap values laters
    I_smooth_interp = np.array([gauss_filterNaN(line,dataPars['windowGCamp']) for line in I])
    G_smooth_interp = np.array([gauss_filterNaN(line,dataPars['windowGCamp']) for line in G])


    print("After smoothing and interpolating", np.nanmean(I_smooth_interp))

    assert np.all(np.isfinite(I_smooth_interp))

    #Reinstate the NaNs
    I_smooth = np.copy(I_smooth_interp)
    I_smooth[np.isnan(I)]=np.nan
    G_smooth = np.copy(G_smooth_interp)
    G_smooth[np.isnan(G)] = np.nan


    #Get the  preferred order  of the neruons,
    #   Jeff did this from Ratio2 in MATLAB:
    #   https://github.com/leiferlab/3dbrain/blob/2c25c6187194263424a0bcfc4d9a0b3b33e31dd9/heatMapGeneration.m#L204
    #       Eventually we want to do our own hierarchical clustering
    #       because this order is based on the ratio which is NOT what we are using.

    jeffsOrdering = False
    if jeffsOrdering:
        order = np.array(data['cgIdx']).T[0] - 1
    else:
        #Don't order things. Keep it simple.
        #NOTE PYTHON INDICES BY ZERO. MATLAB BY ONE.
        order = np.arange(rRaw.shape[0])


    # TODO: Reimplement hierarchical clustering on I, not Ratio and get a new order value

    # Remove flagged Neurons
    badNeurs = np.array([])

    if jeffsOrdering:
        try:
            if len(data['flagged_neurons']) > 0:
                badNeurs = np.array(data['flagged_neurons'][0])
                order = np.delete(order, badNeurs)
        except KeyError:
            pass


    # Identify time points in which the majority of neurons are NaN
    # RATIONALE: For calculations that sum across neurons at a given time
    # (like regression) it doesn't make sense to throw at the whole volume just becase
    # a few neurons are NaNs. So we will interpolate. Conversely, if the whole volume
    # is bad, there is no point including the volume and, if there are large swaths
    # of all NaN timepoints, it could adversely affect our estimate of our regressions'
    # performance.
    #
    # So identify time points that are majority NaN and exclude them.

    #we only allow half the neurons to be NaN. We must correct for the fact that
    # we already no some of the rows are bad so we shouldn't count if they are NaN
    frac_allowed = 0.5 + np.true_divide(badNeurs.shape[0], I.shape[0])
    valid_map = np.mean(np.isnan(I), axis=0) < frac_allowed
    valid_map = np.flatnonzero(valid_map)

    valid_map_data = valid_map[valid_map <= cutVolume]
    valid_map_identity = valid_map[valid_map > cutVolume]

    I_smooth_interp_crop_noncontig_data = np.copy(I_smooth_interp[:, valid_map_data])
    G_smooth_interp_crop_noncontig_data = np.copy(G_smooth_interp[:, valid_map_data])
    I_smooth_interp_crop_noncontig_identity = np.copy(I_smooth_interp[:, valid_map_identity])



    print(np.mean(I_smooth_interp_crop_noncontig_data))
    #<DEPRECATED>

    # load neural data
    RedMatlab = np.array(data['rPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]
    GreenMatlab = np.array(data['gPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]
    #
    Ratio = np.array(data['Ratio2'])[:, :len(np.array(data['hasPointsTime']))]


    Ratio = np.array(data['Ratio2'])[:, :len(np.array(data['hasPointsTime']))]

    # lets interpolate small gaps but throw out larger gaps.
    # make a map with all nans smoothed out if larger than some window
    nanmask = [np.repeat(np.nanmean(chunky_window(line, window=dataPars['interpolateNans']), axis=1),
                         dataPars['interpolateNans']) for line in Ratio]
    nanmask = np.array(nanmask)[:, :Ratio.shape[1]]
    if 'flagged_volumes' in data.keys():
        if len(data['flagged_volumes']) > 0:
            print data['flagged_volumes']
            nanmask[:, np.array(data['flagged_volumes'][0])] = np.nan

    # store relevant indices -- crop out the long gaps of nans adn flagged timepoints
    nonNan  = np.where(np.any(np.isfinite(nanmask),axis=0))[0]
    nonNan_data = nonNan[nonNan <= cutVolume]
    nonNan_identities = nonNan[nonNan > cutVolume]


    #Setup time axis
    time = np.squeeze(data['hasPointsTime'])
    # Set zero to be first non-nan time value
    time -= time[nonNan[0]]
    
    # unpack neuron position (only one frame, randomly chosen)
    try:
        neuroPos = data['XYZcoord'][order].T
    except KeyError:
        neuroPos = []
        print 'No neuron positions:', folder
    
    
    # create a dictionary structure of these data
    dataDict = {}
    # store centerlines subsampled to volumes
    dataDict['CL'] = cl[nonNan]
    dataDict['CLFull'] = cl[idx_data]
    dataDict['goodVolumes'] = nonNan
    dataDict['Behavior'] = {}
    dataDict['BehaviorFull'] = {}
    dataDict['Behavior_crop_noncontig'] = {}
    tmpData = [vel[:,0], pc1, pc2, pc3, vel_ps_downsampled, curv_metric_dowsampled, velo, accel, theta, etho, xPos, yPos]
    for kindex, key in enumerate(['CMSVelocity', 'Eigenworm1', 'Eigenworm2', \
    'Eigenworm3',\
        'PhaseShiftVelocity', 'Curvature',\
                'AngleVelocity', 'AngleAccel', 'Theta', 'Ethogram', 'X', 'Y']):

        dataDict['Behavior'][key] = tmpData[kindex][nonNan_data] #Deprecated

        dataDict['BehaviorFull'][key] = tmpData[kindex][idx_data]
        dataDict['Behavior_crop_noncontig'][key] = tmpData[kindex][valid_map_data]
        dataDict['Behavior_crop_noncontig']['CL'] = cl[valid_map_data]
    dataDict['Behavior']['EthogramFull'] = etho
    dataDict['BehaviorFull']['EthogramFull'] = etho
    dataDict['Neurons'] = {}
    dataDict['Neurons']['Time'] =  time[nonNan_data] # actual time

    #<deprecated>
    dataDict['Neurons']['TimeFull'] =  time[idx_data] # actual time
    dataDict['Neurons']['rRaw'] = (rRaw[:, idx_data])[order,:]
    dataDict['Neurons']['gRaw'] = (gRaw[:, idx_data])[order,:]
    # dataDict['Neurons']['RedRaw'] = RS
    # dataDict['Neurons']['GreenRaw'] = GS

    #</deprecated>

    dataDict['Neurons']['Positions'] = neuroPos
    dataDict['Neurons']['valid'] = nonNan_data
    dataDict['Neurons']['orientation'] = 1 # dorsal or ventral

    dataDict['Neurons']['ordering'] = order

    # Andys improved photobleaching correction, mean- and variance-preserved variables

    dataDict['Neurons']['I'] = I[order][:,idx_data] # linear motion corrected (mean 0, variance preserved) , w/ NaNs,  outlier removed, photobleach corrected
    dataDict['Neurons']['I_Time'] = time[idx_data] #corresponding time axis
    dataDict['Neurons']['I_smooth'] = I_smooth[order][:,idx_data] # SMOOTHED , has nans, linear motion corrected (mean 0, variance preserved) ,, outlier removed, photobleach corrected
    dataDict['Neurons']['I_smooth_interp'] = I_smooth_interp[order][:,idx_data] # interpolated, nans added back in, SMOOTHED,linear motion corrected (mean 0, variance preserved) , outlier removed, photobleach corrected
    dataDict['Neurons']['G'] = G[order][:,idx_data] # outlier removed, photobleach corrected
    dataDict['Neurons']['G_smooth'] = G_smooth[order][:,idx_data] # SMOOTHED has nans,  outlier removed, photobleach corrected
    dataDict['Neurons']['G_smooth_interp'] = G_smooth_interp[order][:,idx_data] # interpolated, nans added back in, SMOOTHED, outlier removed, photobleach corrected

    dataDict['Neurons']['RedMatlab'] = RedMatlab[order] #outlier removed, photobleach corrected
    dataDict['Neurons']['GreenMatlab'] = GreenMatlab[order] #outlier removed, photobleach corrected

    dataDict['Neurons']['I_smooth_interp_crop_noncontig'] = I_smooth_interp_crop_noncontig_data[order] # interpolated, SMOOTHED common noise rejected, mean- and var-preserved, outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict['Neurons']['G_smooth_interp_crop_noncontig'] = G_smooth_interp_crop_noncontig_data[order] # interpolated, SMOOTHED, linear motion correction (mean 0 var preserved), , outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict['Neurons']['I_Time_crop_noncontig'] = time[valid_map_data]  # corresponding time axis
    dataDict['Neurons']['I_valid_map']=valid_map_data

    dataDict['Identities'] = {}
    dataDict['Identities']['rRaw'] = (rRaw[:, idx_identities])[order, :]
    dataDict['Identities']['gRaw'] = (gRaw[:, idx_identities])[order, :]
    return dataDict
    
    
def loadMultipleDatasets(dataLog, pathTemplate, dataPars, nDatasets = None):
    """load matlab files containing brainscanner data. 
    string dataLog: file containing Brainscanner names with timestamps e.g. BrainScanner20160413_133747.
    path pathtemplate: relative or absoluet location of the dataset with a formatter replacing the folder name. e.g.
                        GoldStandardDatasets/{}_linkcopy

    return: dict of dictionaries with neuron and behavior data
    """
    datasets={}
    with open(dataLog, 'r') as f:
        lines = f.readlines()
        for lindex, sline in enumerate(lines):
            sline = sline.strip()
            if not sline or sline[0] == '#':
                continue
            if '#' in sline:
                sline = sline[:sline.index('#')]
            sline = sline.strip()
            line = sline.strip().split(' ')
            folder = ''.join([pathTemplate, line[0], '_MS'])
            if len(line) == 2: #cut volume indicated
                datasets[line[0]] = loadData(folder, dataPars, cutVolume=int(line[1]))
            else:
                datasets[line[0]] = loadData(folder, dataPars)


    return datasets


def gauss_filterNaN(U,sig):
    """Gaussian filter with NaNs. Interpolates also
    This is a slick trick from here:
    https://stackoverflow.com/a/36307291


    Instead of masking, interpolating, and THEN filtering. This
    sets NaNs to zeros, filters, and then normalizes by  dividing off
    the result of a convolution with another mask where nans are 0 and nonNaNs are1.

    It interpolates also.


    One thing I worry about is that this may introduce divide by zero
    errors for large swathso of only NaNs.
    """
    import scipy as sp

    assert np.size(np.shape(U)) == 1, "This function expects one dimensional arrays."

    V = U.copy()
    V[np.isnan(U)] = 0
    VV = sp.ndimage.gaussian_filter1d(V, sigma=sig)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = sp.ndimage.gaussian_filter1d(W, sigma=sig)

    Z_interp = VV / WW

    #For most datasets we are done here. But some datasets have
    #Large swaths of NaNs that are even larger than the window size.
    #In that  case we still have some NaNs floating around.

    valid_mask = np.isfinite(Z_interp)
    invalid_mask = ~valid_mask
    if np.any(invalid_mask) and np.any(valid_mask):
        # If there are still non finite values (like NaNs)
        #Go ahead and do regular old interpolation
        Z_interp[invalid_mask] = np.interp(np.flatnonzero(invalid_mask), np.flatnonzero(valid_mask), Z_interp[valid_mask])
    else:
        Z_interp[invalid_mask] = 0
    return  Z_interp

def loadNeuronPositions(filename):
    x = scipy.io.loadmat(filename)['x']
    y = scipy.io.loadmat(filename)['y']
    z = scipy.io.loadmat(filename)['z']
    neuronID = scipy.io.loadmat(filename)['ID']
    # remove non-head neurons
    indices = np.where((y<-2.3)&(x<0.1))
    return np.stack((neuronID[indices],x[indices],y[indices],z[indices]))




    
def chunky_window(a, window):
    xp =  np.r_[a, np.nan + np.zeros((-len(a) % window,))]
    return xp.reshape(-1, window)
    
def saveDictToHDF(filePath, d):
    f = h5py.File(filePath,'w')
    for fnKey in d.keys(): #this level is datasets ie., Brainscanner0000000
        for amKey in d[fnKey].keys():# this level is analysis type ie., PCA
            if type(d[fnKey][amKey]) is not dict:
                 dataPath = '/%s/%s'%(fnKey,amKey)
                 f.create_dataset(dataPath,data=d[fnKey][amKey])
            else:
                for attKey in d[fnKey][amKey].keys(): # This level is entry ie. PCAweights
                    if type(d[fnKey][amKey][attKey]) is not dict:
                        dataPath = '/%s/%s/%s'%(fnKey,amKey,attKey)
                        f.create_dataset(dataPath,data=d[fnKey][amKey][attKey])
                    else:
                        for bKey in d[fnKey][amKey][attKey].keys():
                            
                            dataPath = '/%s/%s/%s/%s'%(fnKey,amKey,attKey,bKey)
                            f.create_dataset(dataPath,data=d[fnKey][amKey][attKey][bKey])
    f.close()
    return

def loadDictFromHDF(filePath):
    f = h5py.File(filePath,'r')
    d = {}
    for fnKey in f.keys():
        d[fnKey] = {}
        for amKey in f[fnKey].keys():
            if isinstance(f[fnKey][amKey], h5py.Dataset):
                d[fnKey][amKey] = f[fnKey][amKey][...]
            else:
                d[fnKey][amKey] = {}
                for attKey in f[fnKey][amKey].keys():
                    if isinstance(f[fnKey][amKey][attKey], h5py.Dataset):
                        d[fnKey][amKey][attKey] = f[fnKey][amKey][attKey][...]
                    else:
                        d[fnKey][amKey][attKey] = {}
                        for bKey in f[fnKey][amKey][attKey].keys():
                            d[fnKey][amKey][attKey][bKey] = f[fnKey][amKey][attKey][bKey][...]
                        
    f.close()
    return d

def correctPhotobleaching(raw, vps=6, error_bad_fit=False):
    """ Apply photobleaching correction to raw signals, works on a whole heatmap of neurons

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019

    takes raw neural signals N (rows) neruons x T (cols) time points
    we require that each neuron is a row and that there are more time points than rows

    """

    # Make sure the data we are getting makes sense.
    assert raw.ndim <= 2, "raw fluorescent traces passed to correctPhotobleaching() have the wrong number of dimensions"

    if raw.ndim == 2:
        assert raw.shape[1] > raw.shape[
            0], "raw fluorescent traces passed to correctPhotobleaching() have the wrong size! Perhaps it is transposed?"

    window_s = 12.6  # time of median filter window in seconds
    medfilt_window = np.int(np.ceil(window_s * vps / 2.) * 2 + 1)  # make it an odd number of frames

    photoCorr = np.zeros_like(raw)  # initialize photoCorr

    debug = False

    performMedfilt  = True



    if raw.ndim == 2:
        if performMedfilt:
            # perform median filtering (Nan friendly)
            from scipy.signal import medfilt
            smoothed = medfilt(raw, [1, medfilt_window])
        else:
            smoothed = raw.copy()

        N_neurons = raw.shape[0]

        # initialize the exponential fit parameter outputs
        popt = np.zeros([N_neurons, 3])
        pcov = np.zeros([3, 3, N_neurons])

        # Exponential Fit on each neuron
        num_FailsExpFit = 0
        for row in np.arange(N_neurons):
            popt[row, :], pcov[:, :, row], xVals = fitPhotobleaching(smoothed[row, :], vps)

            # If the exponentional we found fits more poorly than a flat line, just use a flat line.
            residual = raw[row, :] - expfunc(xVals, *popt[row, :])  # its ok to have nan's here
            sum_of_squares_fit = np.nansum(np.square(residual))

            flatLine = np.nanmean(raw[row, :])
            residual_flat = raw[row, :] - flatLine
            sum_of_squares_flat = np.nansum(np.square(residual_flat))

            if sum_of_squares_fit > sum_of_squares_flat:
                # The fit does more poorly than a flat line
                #So don't do any photobleaching correction

                print("This trace is better fit by a flat line than an exponential.")
                photoCorr[row, :] = np.copy(raw[row, :])
                showPlots = False
                num_FailsExpFit += 1
            else:
                photoCorr[row, :] = popt[row, 0] * raw[row, :] / expfunc(xVals, *popt[row, :])
                showPlots = False


            if debug:
                if np.random.rand() > 0.85:
                    showPlots = True

            if showPlots:
                import matplotlib.pyplot as plt
                plt.plot(xVals, raw[row, :], 'b-', label=['raw, row: '+np.str(row)])
                plt.plot(xVals, photoCorr[row, :], "r-",
                         label=['photoCorr, row: '+np.str(row)])
                plt.xlabel('Time (s)')
                plt.ylabel('ActivityTrace')
                plt.title('correctPhotobleaching()')
                plt.legend()
                plt.show()

        if np.true_divide(num_FailsExpFit,N_neurons) > 0.5:
            print("Uh oh!: The majority of neurons fail to exhibit exponential decay in their raw signals.\n"+
                    "this could be a sign of a bad recording. \n If this is the red channel its grounds for exclusion., ")
  #          assert error_bad_fit is False, "The majority of neurons fail to exhibit exponential decay in their raw signal compared to flat line."

    elif raw.ndim == 1:
        smoothed = medfilt(raw, medfilt_window)
        popt, pcov, xVals = fitPhotobleaching(smoothed, vps)
        photoCorr= popt[0] * raw / expfunc(xVals, *popt)

    else:
        raise ValueError('The  number dimensions of the data in correctPhotobleaching are not 1 or 2')

    return photoCorr



def fitPhotobleaching(activityTrace, vps):
    """"
    Fit an exponential.

    Accepts a single neuron's activity trace.

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019


    """
    assert activityTrace.ndim == 1, "fitPhotobleaching only works on single neuron traces"

    # Now we will fit an exponential, following along with the tuotiral here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    xVals = np.array(np.arange(activityTrace.shape[-1])) / np.float(vps)

    # Identify just the data that is not a NaN
    nonNaNs = np.logical_not(np.isnan(activityTrace))

    from scipy.optimize import curve_fit

    # set up some bounds on our exponential fitting parameter, y=a*exp(bx)+c
    num_recordingLengths = 8 # the timescale of exponential decay should not be more than eight recording lengths long
    # because otherwise we could have recorded for way longer!!

    #Scale the activity trace to somethign around one.
    scaleFactor=np.nanmean(activityTrace)
    activityTrace=activityTrace/scaleFactor

    bounds = ([0, 1 / (num_recordingLengths * np.nanmax(xVals)), 0],  # lower bounds
              [np.nanmax(activityTrace[nonNaNs])*1.5, 0.5, 2 * np.nanmean(activityTrace)])  # upper bound

    # as a first guess, a is half the max, b is 1/(half the length of the recording), and c is the average
    popt_guess = [np.nanmax(activityTrace) / 2, 2 / np.nanmax(xVals), np.nanmean(activityTrace)]

    popt, pcov = curve_fit(expfunc, xVals[nonNaNs], activityTrace[nonNaNs], p0=popt_guess,
                           bounds=bounds)
    debug = False
    if np.logical_and(np.random.rand() > 0.9, debug):
        showPlots = True
    else:
        showPlots = False

    if showPlots:
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(xVals, activityTrace, 'b-', label='data')
        plt.plot(xVals, expfunc(xVals, *popt), "r-",
                 label='fit a*exp(-b*x)+c: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.xlabel('Time (s)')
        plt.ylabel('ActivityTrace')
        plt.legend()

    ## Now we want to inspect our fit, find values that are clear outliers, and refit while excluding those
    residual = activityTrace - expfunc(xVals, *popt)  # its ok to have nan's here

    nSigmas = 3  # we want to exclude points that are three standard deviations away from the fit

    # Make a new mask that excludes the outliers
    excOutliers = np.copy(nonNaNs)
    excOutliers[(np.abs(residual) > (nSigmas * np.nanstd(residual)))] = False

    # Refit excluding the outliers, use the previous fit as initial guess
    # note we relax the bounds here a bit

    try:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0=popt,bounds=bounds)
    except:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0=popt)

    if showPlots:
        import matplotlib.pyplot as plt
        plt.plot(xVals, expfunc(xVals, *popt), 'y-',
                 label='excluding outliers fit a*exp(-b*x)+c: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.legend()

    #rescale the amplitude, a, back to full size
    popt[0]=scaleFactor*popt[0]

    #rescale the c parameter
    popt[2]=scaleFactor*popt[2]
    return popt, pcov, xVals


def expfunc(x, a, b, c):
    # type: (xVals, a, b, c) -> yVals
    return a * np.exp(-b * x) + c





from skimage.util import view_as_windows

def nanOutliers(data, zscoreBounds = [-2, 5], dataLowerBound = 40, max_nan_per_col = 0.5, vps = 6):
    """Re-implementing Jeff's outlier appraoch from 3DBrain:
    https://github.com/leiferlab/3dbrain/blob/master/heatMapGeneration.m

    """
    dataz=preprocessing.scale(data.T).T

    #nan out the following:

    # intensity values less than 40
    # [ANDY: we shoudl pass this in because for green it should be 0, for red it should be 40]
    data[data < dataLowerBound ] =np.nan


    #anything 5 standard deviations above the mean
    data[dataz > zscoreBounds[1]] = np.nan

    #anything 2 standard deviations below the mean
    # for red this should not be a requirement
    data[dataz < zscoreBounds[0]] = np.nan



    #At the end of finding individual outliers and naning them

    #we need to trash entire columns that have too many nans
    nan_map = np.isnan(data)
    bad_col = np.mean(nan_map, 0) > max_nan_per_col # more than half nans is a bad col

    nan_map[:, bad_col] = True #nan out all of the column




    ## This subsection involves finding dense areas of NaNs and Naning out islands of non-Nans
    ## Which we suspect are unreliable. Some of these comments are taken verbatim from Jeff's Matlab Code

    #do morphological open, removing isolated nans, I'm ok
    #interpolating through some of these
  #  nan_map = scipy.ndimage.binary_opening(nan_map, structure=np.ones((1, 3)))


    # if many nans appear, merge them, so that we don't have isolated values in a sea of nans
   # window = np.int(np.round(vps*1.6666))
    #nan_map = np.logical_or(nan_map, scipy.ndimage.binary_closing(nan_map, structure=np.ones((1, window))))


    #Keep all of the realy bad columns nan'd out
  #  nan_map[:, bad_col] = True #nan out all of the column


    data[ nan_map ] = np.nan



    return data



def close_nan_holes(input):
    import numpy as np
    import scipy as sp
    import scipy.ndimage

    # Function to get rid of isolated NaNs.
    # NaNs are 1.

    mystruct = np.zeros((3, 3))
    mystruct[1, 1] = 1
    mystruct[1, 2] = 1
    mystruct[1, 0] = 1

    a = np.isnan(input)
    b = scipy.ndimage.binary_dilation(a, structure=mystruct).astype(a.dtype)
    c = scipy.ndimage.binary_erosion(b, structure=mystruct).astype(a.dtype)

    out = np.copy(input)
    out[c] = np.nan
    return out

def getCurvature(centerlines):
    ''' Calculate curvature Kappa from the animal's centerline.
    This is a reimplementation of a snipeet of code from Leifer et al Nat Meth 2011
    https://github.com/samuellab/mindcontrol-analysis/blob/cfa4a82b45f0edf204854eed2c68fab337d3be05/preview_wYAML_v9.m#L1555
    Returns curvature in units of inverse worm lengths.

    More information on curvature generally: https://mathworld.wolfram.com/Curvature.html
    Kappa = 1 /R where R is the radius of curvature

    '''
    numcurvpts =np.shape(centerlines)[1]
    diffVec = np.diff(centerlines, axis=1)
    # calculate tangential vectors
    atDiffVec = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
    curv = np.unwrap(np.diff(atDiffVec, axis=1)) # curvature kappa = derivative of angle with respect to path length
                                  # curv = kappa * L/numcurvpts
    curv= curv * numcurvpts     #To get cuvarture in units of 1/L
    return curv

def residualFromShift(dx, template, newframe, inds, roi):
    ''' function takes a new frame of curvature, shifts it with respect to a template and calculates the residual
    dx is the amount of the shift in units of the index (can be fractional)
    teamplate and newframe are curvature vectors with corresponding indices inds
    roi is a list of indexes that defines a smaller ROI of relevant indexes
    '''
    from scipy import interpolate
    shiftfn = interpolate.interp1d(inds, newframe) #function that allows you to shift
    shifted = shiftfn(roi+dx) #look up the values  at the indices specified by the roi shifted by dx
    return template[roi] - shifted  #return the residual

def findPhaseVelocity(curv, clTimes, sigma=50):
    #Find the phase shift
    ps_a0 = findPhaseShift(curv, alpha=0)

    # ps is in units of number of segments per frame
    # we want it in body lengths per second
    NUMSEGS = curv.shape[1] #number of segments per body length
    v = np.append(0, ps_a0[1:] / (NUMSEGS * np.diff(clTimes)))
    # UNITS: (# segs / frame)  / ( (# seconds / frame) * ( # segs / bodylength)    )  = (bodylengths/second)
    # Note the first element of ps_a0 is NOT real, its just always zero because of the way findPhaseVelocity deals with the fact that it is taking a derivative
    # Thats why we have this funny 0 in front
    ps_vel_smooth = gaussian_filter1d(v, sigma=sigma)
    # Note in the future it woudl be better to interpolate onto an even spaced timeline before gaussian smoothing.. but we will ignore that for now.
    return ps_vel_smooth


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/a/6520696/200688

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_curv_metric(curv, ROI_start=15, ROI_end=80, num_stds=6):
    #Calculate mean curvature within an ROI.
    # Identify outliers (default 6 sigma)  and interpolate over them.
    curv_metric = np.mean(curv[:, ROI_start:ROI_end], axis=1)
    #Find outliers & interpolate over them
    outlier_ind = np.where(abs(curv_metric - np.mean(curv_metric)) > num_stds * np.std(curv_metric))
    curv_metric[outlier_ind] = np.nan
    nans, x = nan_helper(curv_metric)
    curv_metric[nans] = np.interp(x(nans), x(~nans), curv_metric[~nans])
    return curv_metric


def debug_curvature_metrics(curv, pc3, num_stds=6):
    #CALCULATE CURVATURE WITHIN ROI
    ROI_start = 15
    ROI_end = 80
    curv_metric = np.mean(curv[:, ROI_start:ROI_end], axis=1)
    #Find outliers & interpolate over them
    outlier_ind = np.where(abs(curv_metric - np.mean(curv_metric)) > num_stds * np.std(curv_metric))
    curv_metric[outlier_ind] = np.nan
    nans, x = nan_helper(curv_metric)
    curv_metric[nans] = np.interp(x(nans), x(~nans), curv_metric[~nans])

    pc3_rescale = pc3 * np.std(curv_metric) / np.std(pc3)
    fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(18,12))
    axs[0].plot(curv_metric, label='Curvature')
    axs[0].plot(pc3_rescale, label='PC3, rescaled')
    axs[0].legend()
    axs[0].set_xlabel('Frames')
    axs[0].set_ylabel('Inverse body lengths')
    axs[1].plot(np.abs(curv_metric - pc3_rescale), label="residual")
    axs[1].legend()
    return

def debug_findPhaseShift(vsmooth, velo, vel, clIndices, clTime, curv, folder=''):
    plt.figure()
    plt.plot(vsmooth)
    from prediction import provenance as prov
    prov.stamp(plt.gca(), .9, .15, __file__)

    plt.figure()
    time = clTime
    plt.plot(time, vsmooth, label= 'phase shift bodylengths/s')
    plt.plot(time, velo * np.std(vsmooth) / np.std(velo), label='eigenworm velo (rescaled)')
    plt.xlabel('Time (s)')
    plt.legend()

    comsmooth = gaussian_filter1d(vel, sigma=6, axis=0)
    norm_eigvel = velo * np.std(vsmooth) / np.std(velo)
    norm_com = comsmooth*np.std(vsmooth) / np.std(comsmooth)

    fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(18,12))
    time = clTime
    axs[0].plot(time, vsmooth, label= 'phase shift')
    axs[0].plot(time, norm_eigvel, label='eigenworm velo (rescaled)')
    axs[0].plot(time[clIndices], norm_com, label='center of mass like (rescaled)')
    axs[0].set_xlim([0, np.max(time[clIndices])])
    axs[0].set_xlabel('Exact Time (s) ')
    axs[0].set_ylabel('Velocity (body lengths / s )')
    axs[0].legend()
    axs[0].set_title(folder)
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(15))

    secax = axs[0].twiny()
    secax.set_xlabel('Centerline Frame')
    mainticks = axs[0].get_xticks()
    secax.set_xticks(np.interp(mainticks, time[clIndices], clIndices))
    axs[1].plot(clIndices, np.std([np.squeeze(vsmooth[clIndices]), np.squeeze(norm_com), np.squeeze(norm_eigvel[clIndices])], axis=0))
    axs[1].set_xlim([0, np.max(clIndices)])
    axs[1].set_xlabel('Centerline Frame')
    axs[1].set_ylabel('Standard Deviation')
    axs[1].set_title('Degree of potential concern')


    plt.figure(figsize=(8,18))
    plt.imshow(curv, vmin=-np.percentile(curv,99), vmax=np.percentile(curv,99), aspect='auto')
    plt.ylabel('Centerline Frame')
    plt.xlabel('Position along Centerlne (<-Head ... Tail->)')
    cax = plt.gca()
    cax.yaxis.set_major_locator(plt.MaxNLocator(30))

    print("done..")
    # plt.show()

    file = os.path.join(folder, "centerline_velocity_analysis.pdf")
    print("Beginning to save centerline debugging plots: " + file)
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(file)
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Centerline debugging plots saved.")
    return

def  findPhaseShift(curv, headCrop=0.2, tailCrop=0.05, alpha=0, sigma=5):
    # Find the phase shift by shifting in x the the n'th frame and comparing it
    # to the (n+1)th frame and recording the shi      ft that gets the best fit.
    #
    # The curvature data is low pass filtered with sigma specified by sigma.
    # Additionally, to filter out noise, the nth frame is averaged with the
    # (n-1)th frame according to a weighting factor, alpha.
    # Moreover the head is cropped a certain porectange according to headCrop,
    # and a percentage of the tail is cropped according to tailCrop.
    #
    # original by Marc Gershow. Modified by Andrew Leifer ca 2010.
    # https://github.com/samuellab/mindcontrol-analysis/blob/master/findPhaseShift.m
    # Python port 2021
    # Marc used to use an alpha of .85 I don't find the weighting useufl.

    #Gaussian filter curvature of each frame
    from scipy.ndimage import gaussian_filter1d
    curvsmooth = gaussian_filter1d(curv,sigma=sigma,axis=1)

    #Get the indices; and cropped indices for only the ROI
    inds = np.arange(curvsmooth.shape[1])
    roi = inds[np.arange(np.floor(curv.shape[1] * headCrop), np.ceil(curv.shape[1]*(1-tailCrop)), dtype=int)]

    from scipy.optimize import least_squares
    from scipy import interpolate
    bounds = [-np.floor(len(inds) * headCrop), np.floor(len(inds) * tailCrop)]
    dx = 0 #initial guess for interframe shift
    ps = np.zeros(curvsmooth.shape[0]) #output
    curvaccumulated = np.copy(curvsmooth[0,:])
    for j, newframe in enumerate(curvsmooth):
        res = least_squares(residualFromShift, np.round(dx, decimals=2), bounds=bounds, args=(curvaccumulated, newframe, inds, roi))
        dx = res.x
        shiftaccum = interpolate.interp1d(inds, curvaccumulated, fill_value="extrapolate")  # function that allows you to shift
        curvaccumulated = alpha * shiftaccum(inds - dx) + (1 - alpha) * newframe #slowly update the new template
        if False: #np.mod(j,1000) == 0:
            plt.figure()
            plt.plot(newframe, label="newframe")
            plt.plot(curvaccumulated, label="curvaccumulated")
            plt.title(dx)
            plt.show()
        #curvaccumulated=np.copy(newframe)
        ps[j] = dx
    return ps





