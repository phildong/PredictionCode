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
from skimage.transform import resize
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit
import h5py
from scipy.special import erf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,  FastICA

import userTracker

import logging
logger = logging.getLogger('dataHandler')
logger.setLevel(level = logging.ERROR)

def deconvolveCalcium(Y, show = False):
    """deconvolve with GCamp6s response digitized from Nature volume 499, pages 295–300 (18 July 2013)
        doi:10.1038/nature12354"""
    # fit function -- fitted with least squares from digitized data

    pars =  [0.38036106, 0.00565365, 1.00621729, 0.31627363 ]
    def fitfunc(x,A,m, tau1, s):
        return A*erf((x-m)/s)*np.exp(-x/tau1)

    gcampXN = np.linspace(0,Y.shape[1]/6., Y.shape[1])
    gcampYN = fitfunc(gcampXN, *pars)

    Ydec = np.real(np.fft.ifft(np.fft.fft(Y, axis = 1)/np.fft.fft(gcampYN)))*np.sum(gcampYN)
    
    return Ydec

def makeEthogram(anglevelocity, pc3):
    """use rotated Eigenworms to create a new Ethogram."""

    etho = np.zeros((len(anglevelocity),1))

    # set forward and backward
    etho[np.where(anglevelocity>0.05)] = 1
    etho[np.where(anglevelocity<=-0.05)] = -1

    # overwrite this in case of turns
    etho[np.abs(pc3)>10] = 2

    # mask nans in ethogram
    ethomask = np.isnan(etho)
    if np.any(ethomask):
        etho[ethomask] = 0

    return etho

def loadPoints(folder, straight = True):
    """get tracked points from Pointfile."""
    points = np.squeeze(scipy.io.loadmat(os.path.join(folder,'pointStatsNew.mat'))['pointStatsNew'])
    
    if straight:
        return [p[1] for p in points]
    else:
        return [p[2] for p in points]
    
def loadEigenBasis(filename, nComp = 3, new = True):
    """load the specific worm basis set."""
    if new:
        eigenworms = np.loadtxt(filename)[:nComp]
    else:
        eigenworms = scipy.io.loadmat(filename)['eigbasis'][:nComp]

    # take first nComp eigenworms
    eigenworms = resize(eigenworms, (nComp,99))

    return eigenworms

def calculateEigenwormsFromCL(cl, eigenworms):
    """takes (x,y) pairs from centerlines and returns eigenworm coefficients."""

    # calculate tangential vectors
    diffVec = np.diff(cl, axis = 1)
    wcNew = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
    
    # calculate mean subtracted tangent angles
    meanAngle = np.mean(wcNew, axis = 1)
    wcNew = wcNew-meanAngle[:, np.newaxis]
    
    # project onto Eigenworms
    pcsNew = np.dot(eigenworms,wcNew.T)

    # compute segment lengths, reference point needed for reconstruction
    lengths = np.sqrt(diffVec[:,:,1]**2+diffVec[:,:,0]**2)
    refPoint = cl[:,0]

    return pcsNew, meanAngle, lengths, refPoint

def loadCenterlines(folder, full = False, wormcentered = False):
    """get centerlines from centerline.mat file"""
    
    # load all centerline data
    if wormcentered:
        cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['wormcentered'], 1, 0)
    else:
        cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['centerline'], 2, 0)
    
    # load timing for centerlines and neurons
    heatData = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    clTime = np.squeeze(heatData['clTime']) # 50Hz centerline times
    volTime =  np.squeeze(heatData['hasPointsTime']) # 6 vol/sec neuron times
    
    # sync up centerlines with neurons
    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime)))).astype(int)
    if not full:
        cl = cl[clIndices]
        
    return cl, clIndices
    
def transformEigenworms(pcs, dataPars):
    """interpolate, smooth Eigenworms and calculate associated metrics like velocity."""

    pc3, pc2, pc1 = pcs

    # Mask nans in eigenworms by linear interpolation
    for pcindex, pc in enumerate(pcs):
        mask = np.isnan(pc1)

        if np.any(mask):
            pc[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pc1[~mask])
            logger.warning('Found ' + str(np.sum(mask)) + 'NaNs that are being interpolated over in the behavior PC1, PC2 or PC3.')
        
        pcs[pcindex] = pc
        
    theta = np.unwrap(np.arctan2(pcs[2], pcs[1]))
    # compute first and second derivatives of posture angle theta
    velo = gaussian_filter1d(theta, dataPars['gaussWindow'], order = 1) # radians/frame
    accel = gaussian_filter1d(theta, 2*dataPars['gaussWindow'], order = 2)

    return pcs, theta, velo,  accel


def decorrelateNeuronsICA(R, G):
    """
    Use Independent Component Analysis (ICA) to remove covariance in Green and Red signals.
    NaN values are excluded. Variance and mean is preserved.
    """

    # initialize independent component I with NaNs
    I = np.empty(G.shape)
    I[:] = np.nan

    ica = FastICA(n_components = 2)
    scaler = StandardScaler()

    # mask to exclude NaNs before running the ICA
    nanmask = np.logical_or(np.isnan(R), np.isnan(G))

    for li in range(len(R)):
        Y = np.vstack([R[li, ~nanmask[li]], G[li, ~nanmask[li]]]).T

        if Y.size <= 2: # if we don't have enough non-NaNs, skip ICA
            continue

        S = ica.fit_transform(scaler.fit_transform(Y))

        # find component with minimal correlation to red signal
        idx = np.argmin(np.abs([np.corrcoef(s, R[li, ~nanmask[li]])[0, 1] for s in S.T]))
        # invert component if necessary
        sign = np.sign(np.corrcoef(S[:, idx], G[li, ~nanmask[li]])[0, 1])
        signal = sign * (S[:, idx])

        # rescale and add back mean. NB: this means that I values can be negative
        rescaledSignalTrunc = (np.std(G[li, ~nanmask[li]]) / np.std(signal)) * signal + np.mean(G[li, ~nanmask[li]])

        # store calculated signal
        I[li, ~nanmask[li]] = rescaledSignalTrunc
    
    return I

def loadData(folder, dataPars, ew = 1, cutVolume = None):
    """Load neural and behavior data from Matlab files"""
    logger.info('Loading ', folder)
    try:
        data = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    except IOError:
        logger.error('heatDataMS.mat not found, attempting to load heatData.mat')
        data = scipy.io.loadmat(os.path.join(folder,'heatData.mat'))

    # unpack behavior variables
    ethoOrig, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T

    # get centerlines with full temporal resolution of 50Hz
    clFull, clIndices = loadCenterlines(folder, full = True)

    # load eigenworms
    eigenworms = loadEigenBasis(filename = os.path.join(userTracker.codePath(), 'utility/Eigenworms.dat'), nComp = 3, new = True)

    # project postures onto eigenworms
    pcsFull, meanAngle, lengths, refPoint = calculateEigenwormsFromCL(clFull, eigenworms)

    # do Eigenworm transformations and calculate velocity etc. 
    pcs, theta, velo, accel = transformEigenworms(pcsFull, dataPars)

    # downsample pcs, centerline, posture to 6 volumes/sec
    pc3, pc2, pc1 = pcs[:,clIndices]
    theta = theta[clIndices]
    cl = clFull[clIndices]

    # downsample and convert velocity and acceleration to radians/sec
    velo = velo[clIndices]*50. 
    accel = accel[clIndices]*50

    etho = makeEthogram(velo, pc3)

    rRaw = np.array(data['rRaw'])[:,:len(np.array(data['hasPointsTime']))]
    gRaw = np.array(data['gRaw'])[:,:len(np.array(data['hasPointsTime']))]

    # if a cut volume is defined, split into data (before the cut volume) and AVA identification phase (after the cut volume)
    if cutVolume is None:
        cutVolume = rRaw.shape[1]

    idx = np.arange(rRaw.shape[1])
    idx_data = idx[idx <= cutVolume]
    idx_identities = idx[idx > cutVolume]

    rphotocorr = np.array(data['rPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]
    gphotocorr = np.array(data['gPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]

    # load neural data
    R = rRaw.copy()[:, idx_data]
    G = gRaw.copy()[:, idx_data]

    vps = dataPars['volumeAcquisitionRate']

    R = correctPhotobleaching(R, vps, error_bad_fit = True)
    G = correctPhotobleaching(G, vps)

    R[np.isnan(rphotocorr[:, idx_data])] = np.nan
    G[np.isnan(gphotocorr[:, idx_data])] = np.nan

    # Remove isolated non-NaNs in a sea of NaNs
    R = close_nan_holes(R)
    G = close_nan_holes(G)

    # NaN-out volumes which were manually flagged
    if 'flagged_volumes' in data.keys():
        if len(data['flagged_volumes']) > 0:
            R[:, np.array(data['flagged_volumes'][0])] = np.nan
            G[:, np.array(data['flagged_volumes'][0])] = np.nan

    # Reject noise common to both Red and Green Channels using ICA
    I = decorrelateNeuronsICA(R, G)

    # Apply Gaussian Smoothing (and interpolate for free)
    I_smooth_interp = np.array([gauss_filterNaN(line,dataPars['windowGCamp']) for line in I])

    assert np.all(np.isfinite(I_smooth_interp))

    # Reinstate the NaNs
    I_smooth = np.copy(I_smooth_interp)
    I_smooth[np.isnan(I)] = np.nan


    # Identify time points in which the majority of neurons are NaN
    # RATIONALE: For calculations that sum across neurons at a given time
    # (like regression) it doesn't make sense to throw at the whole volume just becase
    # a few neurons are NaNs. So we will interpolate. Conversely, if the whole volume
    # is bad, there is no point including the volume and, if there are large swaths
    # of all NaN timepoints, it could adversely affect our estimate of our regression
    # performance.
    #
    # So identify time points that are majority NaN and exclude them.

    valid_map = np.mean(np.isnan(I), axis = 0) < 0.5
    valid_map = np.flatnonzero(valid_map)

    valid_map_data = valid_map[valid_map <= cutVolume]
    valid_map_identity = valid_map[valid_map > cutVolume]

    I_smooth_interp_crop_noncontig_data = np.copy(I_smooth_interp[:, valid_map_data])
    I_smooth_interp_crop_noncontig_identity = np.copy(I_smooth_interp[:, valid_map_identity])

    # Setup time axis
    time = np.squeeze(data['hasPointsTime'])
    # Set zero to be first non-nan time value
    nonNan  = np.where(np.any(np.isfinite(I_smooth),axis = 0))[0]
    nonNan_data = nonNan[nonNan <= cutVolume]
    nonNan_identities = nonNan[nonNan > cutVolume]
    time -= time[nonNan[0]]
    
    # unpack neuron position (only one frame, randomly chosen)
    try:
        neuroPos = data['XYZcoord'].T
    except KeyError:
        neuroPos = []
        logger.error('No neuron positions found:', folder)
    
    
    # create a dictionary to store all the data
    dataDict = {}
    # store centerlines subsampled to volumes
    dataDict['CL'] = cl[nonNan]
    dataDict['CLFull'] = cl[idx_data]
    # store indices of non-NaN volumes
    dataDict['goodVolumes'] = nonNan

    # store behavior data
    dataDict['Behavior'] = {}
    dataDict['BehaviorFull'] = {}
    dataDict['Behavior_crop_noncontig'] = {}
    tmpData = [vel[:,0], pc1, pc2, pc3, velo, accel, theta, etho, xPos, yPos]
    for kindex, key in enumerate(['CMSVelocity', 'Eigenworm1', 'Eigenworm2', 'Eigenworm3', \
                                  'AngleVelocity', 'AngleAccel', 'Theta', 'Ethogram', 'X', 'Y']):
        dataDict['Behavior'][key] = tmpData[kindex][nonNan_data]
        dataDict['BehaviorFull'][key] = tmpData[kindex][idx_data]
        dataDict['Behavior_crop_noncontig'][key] = tmpData[kindex][valid_map_data]

    dataDict['Behavior_crop_noncontig']['CL'] = cl[valid_map_data]    
    dataDict['Behavior']['EthogramFull'] = etho
    dataDict['BehaviorFull']['EthogramFull'] = etho

    # store neuron data
    dataDict['Neurons'] = {}
    dataDict['Neurons']['Time'] =  time[nonNan_data]
    dataDict['Neurons']['Positions'] = neuroPos
    dataDict['Neurons']['valid'] = nonNan_data
    dataDict['Neurons']['orientation'] = 1 # dorsal or ventral

    # Andys improved photobleaching correction, mean- and variance-preserved variables

    dataDict['Neurons']['I'] = I[:,idx_data] # common noise rejected, w/ NaNs, mean- and var-preserved, outlier removed, photobleach corrected
    dataDict['Neurons']['I_Time'] = time[idx_data] #corresponding time axis
    dataDict['Neurons']['I_smooth'] = I_smooth[:,idx_data] # SMOOTHED common noise rejected, has nans, mean- and var-preserved, outlier removed, photobleach corrected
    dataDict['Neurons']['I_smooth_interp'] = I_smooth_interp[:,idx_data] # interpolated, nans added back in, SMOOTHED common noise rejected, mean- and var-preserved, outlier removed, photobleach corrected
    dataDict['Neurons']['R'] = R #outlier removed, photobleach corrected
    dataDict['Neurons']['G'] = G #outlier removed, photobleach corrected

    dataDict['Neurons']['I_smooth_interp_crop_noncontig'] = I_smooth_interp_crop_noncontig_data # interpolated, SMOOTHED common noise rejected, mean- and var-preserved, outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict['Neurons']['I_Time_crop_noncontig'] = time[valid_map_data]  # corresponding time axis
    dataDict['Neurons']['I_valid_map'] = valid_map_data

    dataDict['Identities'] = {}
    dataDict['Identities']['rRaw'] = (rRaw[:, idx_identities])
    dataDict['Identities']['gRaw'] = (gRaw[:, idx_identities])

    return dataDict
    
    
def loadMultipleDatasets(dataLog, pathTemplate, dataPars):
    """
    Load Matlab files containing BrainScanner data. 
    
    dataLog: name of file containing BrainScanner names with timestamps e.g. AML32_moving/AML32_moving_datasets.txt
    pathtemplate: relative or absolute location of the dataset, e.g. AML32_moving/

    return: dict of dictionaries with neuron and behavior data
    """
    datasets = {}
    with open(dataLog, 'r') as f:
        lines = f.readlines()
        for lindex, sline in enumerate(lines):
            sline = sline.strip()
            if not sline or sline[0] == '#':
                continue
            if '#' in sline:
                sline = sline[:sline.index('#')].strip()
            line = sline.split(' ')
            folder = ''.join([pathTemplate, line[0], '_MS'])

            if len(line) == 2: # cut volume indicated
                datasets[line[0]] = loadData(folder, dataPars, cutVolume = int(line[1]))
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
    """
    import scipy as sp

    assert np.size(np.shape(U)) == 1, "gauss_filterNaN expects one dimensional arrays."

    V = U.copy()
    V[np.isnan(U)] = 0
    VV = sp.ndimage.gaussian_filter1d(V, sigma = sig)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = sp.ndimage.gaussian_filter1d(W, sigma = sig)

    Z_interp = VV / WW

    # For most datasets we are done here. But some datasets have
    # large swaths of NaNs that are even larger than the window size.
    # In that case we still have some NaNs floating around.

    valid_mask = np.isfinite(Z_interp)
    invalid_mask = ~valid_mask
    if np.any(invalid_mask) and np.any(valid_mask):
        # If there are still non finite values (like NaNs), go ahead and do regular old interpolation
        Z_interp[invalid_mask] = np.interp(np.flatnonzero(invalid_mask), np.flatnonzero(valid_mask), Z_interp[valid_mask])
    else:
        Z_interp[invalid_mask] = 0
    
    return Z_interp

def expfunc(x, a, b, c): 
    return a * np.exp(-b * x) + c

def correctPhotobleaching(raw, vps = 6, error_bad_fit = False):
    """ 
    Apply photobleaching correction to raw signals, works on a whole heatmap of neurons

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019

    takes raw neural signals N (rows) neurons x T (cols) time points
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

    if raw.ndim == 2:
        # perform median filtering (Nan friendly)
        from scipy.signal import medfilt
        smoothed = medfilt(raw, [1, medfilt_window])

        N_neurons = raw.shape[0]

        # initialize the exponential fit parameter outputs
        popt = np.zeros([N_neurons, 3])

        # Exponential Fit on each neuron
        num_FailsExpFit = 0
        for row in np.arange(N_neurons):
            popt[row, :], _, xVals = fitPhotobleaching(smoothed[row, :], vps)

            # Compare residuals to that of a flat line
            residual = raw[row, :] - expfunc(xVals, *popt[row, :])
            sum_of_squares_fit = np.nansum(np.square(residual))

            flatLine = np.nanmean(raw[row, :])
            residual_flat = raw[row, :] - flatLine
            sum_of_squares_flat = np.nansum(np.square(residual_flat))

            # If the fit does more poorly than a flat line, don't do any photobleaching correction
            if sum_of_squares_fit > sum_of_squares_flat:
                logger.warning("This trace is better fit by a flat line than an exponential.")
                photoCorr[row, :] = np.copy(raw[row, :])
                num_FailsExpFit += 1
            else:
                photoCorr[row, :] = popt[row, 0] * raw[row, :] / expfunc(xVals, *popt[row, :])

        if np.true_divide(num_FailsExpFit,N_neurons) > 0.5:
            logger.error("Uh oh!: The majority of neurons fail to exhibit exponential decay in their raw signals.\n \
                             This could be a sign of a bad recording. \n If this is the red channel, it is grounds for exclusion.")

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

    xVals = np.array(np.arange(activityTrace.shape[-1])) / np.float(vps)

    nonNaNs = np.logical_not(np.isnan(activityTrace))

    # set up some bounds on our exponential fitting parameter, y = a*exp(bx)+c
    # the timescale of exponential decay should not be more than eight recording lengths long
    # because otherwise we could have recorded for way longer!!
    num_recordingLengths = 8

    # Scale the activity trace to somethign around one.
    scaleFactor = np.nanmean(activityTrace)
    activityTrace = activityTrace / scaleFactor

    bounds = ([0, 1 / (num_recordingLengths * np.nanmax(xVals)), 0],  # lower bounds
              [np.nanmax(activityTrace[nonNaNs])*1.5, 0.5, 2 * np.nanmean(activityTrace)])  # upper bound

    # as a first guess, a is half the max, b is 1/(half the length of the recording), and c is the average
    popt_guess = [np.nanmax(activityTrace) / 2, 2 / np.nanmax(xVals), np.nanmean(activityTrace)]

    popt, pcov = curve_fit(expfunc, xVals[nonNaNs], activityTrace[nonNaNs], p0 = popt_guess, bounds = bounds)

    # Now we want to inspect our fit, find values that are clear outliers, and refit while excluding those
    residual = activityTrace - expfunc(xVals, *popt)

    nSigmas = 3  # we want to exclude points that are three standard deviations away from the fit

    # Make a new mask that excludes the outliers
    excOutliers = np.copy(nonNaNs)
    excOutliers[(np.abs(residual) > (nSigmas * np.nanstd(residual)))] = False

    # Refit excluding the outliers, use the previous fit as initial guess
    # note we relax the bounds here a bit

    try:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0 = popt, bounds = bounds)
    except:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0 = popt)

    #rescale the amplitude, a, back to full size
    popt[0] = scaleFactor*popt[0]

    #rescale the c parameter
    popt[2] = scaleFactor*popt[2]

    return popt, pcov, xVals

def close_nan_holes(input):
    """
    Remove isolated NaNs
    """

    mystruct = np.zeros((3, 3))
    mystruct[1, :] = 1

    a = np.isnan(input)
    b = scipy.ndimage.binary_dilation(a, structure = mystruct).astype(a.dtype)
    c = scipy.ndimage.binary_erosion(b, structure = mystruct).astype(a.dtype)

    out = np.copy(input)
    out[c] = np.nan

    return out