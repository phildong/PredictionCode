# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:10 2017
load and compare multiple data sets for behavior prediction.
@author: monika scholz
"""
# make division give you floats, as is standard for python3

import os

import numpy as np
import scipy.interpolate
import scipy.io
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import erf
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


def loadPoints(folder, straight=True):
    """get tracked points from Pointfile."""
    points = np.squeeze(
        scipy.io.loadmat(os.path.join(folder, "pointStatsNew.mat"))["pointStatsNew"]
    )

    if straight:
        return [p[1] for p in points]
    else:
        return [p[2] for p in points]


def loadCenterlines(folder, full=False, wormcentered=False):
    """get centerlines from centerline.mat file"""
    tmp = scipy.io.loadmat(os.path.join(folder, "centerline.mat"))

    if wormcentered:
        cl = np.rollaxis(tmp["wormcentered"], 1, 0)
    else:
        cl = np.rollaxis(tmp["centerline"], 2, 0)

    tmp = scipy.io.loadmat(os.path.join(folder, "heatDataMS.mat"))

    clTime = np.squeeze(tmp["clTime"])  # 50Hz centerline times
    volTime = np.squeeze(tmp["hasPointsTime"])  # 6 vol/sec neuron times

    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime))))
    if not full:
        cl = cl[clIndices.astype(int)]

    return cl, clIndices.astype(int), clTime


def decorrelateNeuronsLinear(R, G):
    # Xinwei Yu's linear motion correction algorithm
    # Re-implemented by Andrew Leifer, based on an implimentation by Matthew Creamer
    I = np.empty(G.shape)
    I[:] = np.nan  # initialize it with nans

    # the fits can't handle the nans, so we gotta tempororaily exclude them
    # we won't interpolate, we will just remove them than add them back
    nanmask = np.logical_or(np.isnan(R), np.isnan(G))

    for n in range(R.shape[0]):  # for each neuron
        red_column = np.expand_dims(R[n, ~nanmask[n]].T, axis=1)
        red_with_ones = np.concatenate([red_column, np.ones(red_column.shape)], axis=1)
        best_fit, _, _, _ = np.linalg.lstsq(
            red_with_ones, G[n, ~nanmask[n]].T, rcond=None
        )
        I[n, :] = G[n, :] - (best_fit[0] * R[n, :] + best_fit[1])

    return I


def loadData(folder, dataPars, ew=1, cutVolume=None):
    """load matlab data."""
    print("Loading ", folder)
    try:
        data = scipy.io.loadmat(os.path.join(folder, "heatDataMS.mat"))
    except IOError:
        print("IOERROR")
        data = scipy.io.loadmat(os.path.join(folder, "heatData.mat"))

    # unpack behavior variables
    ethoOrig, xPos, yPos, vel, pc12, pc3 = data["behavior"][0][0].T

    if cutVolume is None:
        cutVolume = np.max(vel.size)

    # get centerlines with full temporal resolution of 50Hz
    clFull, clIndices, clTimes = loadCenterlines(folder, full=True)
    curv = getCurvature(clFull)

    curv_metric = get_curv_metric(curv, ROI_start=15, ROI_end=80, num_stds=6)

    # downsample to 6 volumes/sec
    curv_metric_downsampled = curv_metric[clIndices]

    rRaw = np.array(data["rRaw"])[:, : len(np.array(data["hasPointsTime"]))]
    gRaw = np.array(data["gRaw"])[:, : len(np.array(data["hasPointsTime"]))]

    idx = np.arange(rRaw.shape[1])
    idx_data = idx[idx <= cutVolume]
    idx_identities = idx[idx > cutVolume]

    rphotocorr = np.array(data["rPhotoCorr"])[:, : len(np.array(data["hasPointsTime"]))]
    gphotocorr = np.array(data["gPhotoCorr"])[:, : len(np.array(data["hasPointsTime"]))]

    # load neural data
    R = rRaw.copy()[:, idx_data]
    G = gRaw.copy()[:, idx_data]

    vps = dataPars["volumeAcquisitionRate"]

    R = correctPhotobleaching(R, vps, error_bad_fit=True)
    G = correctPhotobleaching(G, vps)

    # Below we apply Jeff's outlier nan'ing
    R[np.isnan(rphotocorr[:, idx_data])] = np.nan
    G[np.isnan(gphotocorr[:, idx_data])] = np.nan

    # Remove isolated non-NaNs in a sea of NaNs
    R = close_nan_holes(R)
    G = close_nan_holes(G)

    # NaN-out Flagged volumes (presumably these are manually flagged somewhere in the pipeline)
    if "flagged_volumes" in list(data.keys()):
        if len(data["flagged_volumes"]) > 0:
            R[:, np.array(data["flagged_volumes"][0])] = np.nan
            G[:, np.array(data["flagged_volumes"][0])] = np.nan

    # Reject noise common to both Red and Green Channels using linear motion correction algorithm
    I = decorrelateNeuronsLinear(R, G)

    # Apply Gaussian Smoothing (and interpolate for free) #Keep the raw green around so we can use it to set heatmap values laters
    I_smooth_interp = np.array(
        [gauss_filterNaN(line, dataPars["windowGCamp"]) for line in I]
    )
    G_smooth_interp = np.array(
        [gauss_filterNaN(line, dataPars["windowGCamp"]) for line in G]
    )
    R_smooth_interp = np.array(
        [gauss_filterNaN(line, dataPars["windowGCamp"]) for line in R]
    )

    assert np.all(np.isfinite(I_smooth_interp))

    # Reinstate the NaNs
    I_smooth = np.copy(I_smooth_interp)
    I_smooth[np.isnan(I)] = np.nan
    G_smooth = np.copy(G_smooth_interp)
    G_smooth[np.isnan(G)] = np.nan
    R_smooth = np.copy(R_smooth_interp)
    R_smooth[np.isnan(R)] = np.nan

    # Get the  preferred order  of the neruons,
    #   Jeff did this from Ratio2 in MATLAB:
    #   https://github.com/leiferlab/3dbrain/blob/2c25c6187194263424a0bcfc4d9a0b3b33e31dd9/heatMapGeneration.m#L204
    #       We want to do our own hierarchical clustering
    #       because this order is based on the ratio which is NOT what we are using.

    order = np.arange(rRaw.shape[0])
    # TODO: Reimplement hierarchical clustering on I, not Ratio and get a new order value

    # Remove flagged Neurons
    badNeurs = np.array([])

    # Identify time points in which the majority of neurons are NaN
    # RATIONALE: For calculations that sum across neurons at a given time
    # (like regression) it doesn't make sense to throw at the whole volume just becase
    # a few neurons are NaNs. So we will interpolate. Conversely, if the whole volume
    # is bad, there is no point including the volume and, if there are large swaths
    # of all NaN timepoints, it could adversely affect our estimate of our regressions'
    # performance.
    #
    # So identify time points that are majority NaN and exclude them.

    # we only allow half the neurons to be NaN. We must correct for the fact that
    # we already no some of the rows are bad so we shouldn't count if they are NaN
    frac_allowed = 0.5 + np.true_divide(badNeurs.shape[0], I.shape[0])
    valid_map = np.mean(np.isnan(I), axis=0) < frac_allowed
    valid_map = np.flatnonzero(valid_map)

    valid_map_data = valid_map[valid_map <= cutVolume]
    valid_map_identity = valid_map[valid_map > cutVolume]

    I_smooth_interp_crop_noncontig_data = np.copy(I_smooth_interp[:, valid_map_data])
    G_smooth_interp_crop_noncontig_data = np.copy(G_smooth_interp[:, valid_map_data])
    R_smooth_interp_crop_noncontig_data = np.copy(R_smooth_interp[:, valid_map_data])
    I_smooth_interp_crop_noncontig_identity = np.copy(
        I_smooth_interp[:, valid_map_identity]
    )

    RedMatlab = np.array(data["rPhotoCorr"])[:, : len(np.array(data["hasPointsTime"]))]
    GreenMatlab = np.array(data["gPhotoCorr"])[
        :, : len(np.array(data["hasPointsTime"]))
    ]

    # Setup time axis
    time = np.squeeze(data["hasPointsTime"])
    # Set zero to be first non-nan time value
    time -= time[valid_map_data[0]]

    # unpack neuron position (only one frame, randomly chosen)
    try:
        neuroPos = data["XYZcoord"][order].T
    except KeyError:
        neuroPos = []
        print("No neuron positions:", folder)

    # create a dictionary structure of these data
    dataDict = {}
    dataDict["BehaviorFull"] = {}
    dataDict["Behavior_crop_noncontig"] = {}

    tmpData = [vel[:, 0], curv_metric_downsampled, xPos, yPos]
    for kindex, key in enumerate(["CMSVelocity", "Curvature", "X", "Y"]):
        dataDict["BehaviorFull"][key] = tmpData[kindex][idx_data]
        dataDict["Behavior_crop_noncontig"][key] = tmpData[kindex][valid_map_data]

    dataDict["Neurons"] = {}

    dataDict["Neurons"]["TimeFull"] = time[idx_data]  # actual time
    dataDict["Neurons"]["rRaw"] = (rRaw[:, idx_data])[order, :]
    dataDict["Neurons"]["gRaw"] = (gRaw[:, idx_data])[order, :]

    dataDict["Neurons"]["Positions"] = neuroPos
    dataDict["Neurons"]["ordering"] = order

    # Andys improved photobleaching correction, mean- and variance-preserved variables

    dataDict["Neurons"]["I"] = I[order][
        :, idx_data
    ]  # linear motion corrected (mean 0, variance preserved) , w/ NaNs,  outlier removed, photobleach corrected
    dataDict["Neurons"]["I_Time"] = time[idx_data]  # corresponding time axis
    dataDict["Neurons"]["I_smooth"] = I_smooth[order][
        :, idx_data
    ]  # SMOOTHED , has nans, linear motion corrected (mean 0, variance preserved) ,, outlier removed, photobleach corrected
    dataDict["Neurons"]["I_smooth_interp"] = I_smooth_interp[order][
        :, idx_data
    ]  # interpolated, nans added back in, SMOOTHED,linear motion corrected (mean 0, variance preserved) , outlier removed, photobleach corrected
    dataDict["Neurons"]["G"] = G[order][
        :, idx_data
    ]  # outlier removed, photobleach corrected
    dataDict["Neurons"]["G_smooth"] = G_smooth[order][
        :, idx_data
    ]  # SMOOTHED has nans,  outlier removed, photobleach corrected
    dataDict["Neurons"]["G_smooth_interp"] = G_smooth_interp[order][
        :, idx_data
    ]  # interpolated, nans added back in, SMOOTHED, outlier removed, photobleach corrected
    dataDict["Neurons"]["R"] = R[order][
        :, idx_data
    ]  # outlier removed, photobleach corrected
    dataDict["Neurons"]["R_smooth"] = R_smooth[order][
        :, idx_data
    ]  # SMOOTHED has nans,  outlier removed, photobleach corrected
    dataDict["Neurons"]["R_smooth_interp"] = R_smooth_interp[order][
        :, idx_data
    ]  # interpolated, nans added back in, SMOOTHED, outlier removed, photobleach corrected

    dataDict["Neurons"]["RedMatlab"] = RedMatlab[
        order
    ]  # outlier removed, photobleach corrected
    dataDict["Neurons"]["GreenMatlab"] = GreenMatlab[
        order
    ]  # outlier removed, photobleach corrected

    dataDict["Neurons"][
        "I_smooth_interp_crop_noncontig"
    ] = I_smooth_interp_crop_noncontig_data[
        order
    ]  # interpolated, SMOOTHED common noise rejected, mean- and var-preserved, outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict["Neurons"][
        "G_smooth_interp_crop_noncontig"
    ] = G_smooth_interp_crop_noncontig_data[
        order
    ]  # interpolated, SMOOTHED, linear motion correction (mean 0 var preserved), , outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict["Neurons"][
        "R_smooth_interp_crop_noncontig"
    ] = R_smooth_interp_crop_noncontig_data[
        order
    ]  # interpolated, SMOOTHED, linear motion correction (mean 0 var preserved), , outlier removed, photobleach corrected, note strings of nans have been removed such that the DeltaT between elements is no longer constant
    dataDict["Neurons"]["I_Time_crop_noncontig"] = time[
        valid_map_data
    ]  # corresponding time axis
    dataDict["Neurons"]["I_valid_map"] = valid_map_data

    dataDict["Identities"] = {}
    dataDict["Identities"]["rRaw"] = (rRaw[:, idx_identities])[order, :]
    dataDict["Identities"]["gRaw"] = (gRaw[:, idx_identities])[order, :]

    return dataDict


def loadMultipleDatasets(dataLog, pathTemplate, nDatasets=None):
    """load matlab files containing brainscanner data.
    string dataLog: file containing Brainscanner names with timestamps e.g. BrainScanner20160413_133747.
    path pathtemplate: relative or absoluet location of the dataset with a formatter replacing the folder name. e.g.
                        GoldStandardDatasets/{}_linkcopy

    return: dict of dictionaries with neuron and behavior data
    """

    datasets = {}
    dataPars = {
        "windowGCamp": 5,  # gauss window for red and green channel
        "volumeAcquisitionRate": 6.0,  # rate at which volumes are acquired
    }

    with open(dataLog, "r") as f:
        lines = f.readlines()
        for lindex, sline in enumerate(lines):
            sline = sline.strip()
            if not sline or sline[0] == "#":
                continue
            if "#" in sline:
                sline = sline[: sline.index("#")]
            sline = sline.strip()
            line = sline.strip().split(" ")
            folder = "".join([pathTemplate, line[0], "_MS"])
            if len(line) == 2:  # cut volume indicated
                datasets[line[0]] = loadData(folder, dataPars, cutVolume=int(line[1]))
            else:
                datasets[line[0]] = loadData(folder, dataPars)

    return datasets


def gauss_filterNaN(U, sig):
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

    with np.errstate(invalid="ignore"):
        Z_interp = VV / WW

    # For most datasets we are done here. But some datasets have
    # Large swaths of NaNs that are even larger than the window size.
    # In that  case we still have some NaNs floating around.

    valid_mask = np.isfinite(Z_interp)
    invalid_mask = ~valid_mask
    if np.any(invalid_mask) and np.any(valid_mask):
        # If there are still non finite values (like NaNs)
        # Go ahead and do regular old interpolation
        Z_interp[invalid_mask] = np.interp(
            np.flatnonzero(invalid_mask),
            np.flatnonzero(valid_mask),
            Z_interp[valid_mask],
        )
    else:
        Z_interp[invalid_mask] = 0
    return Z_interp


def loadNeuronPositions(filename):
    x = scipy.io.loadmat(filename)["x"]
    y = scipy.io.loadmat(filename)["y"]
    z = scipy.io.loadmat(filename)["z"]
    neuronID = scipy.io.loadmat(filename)["ID"]
    # remove non-head neurons
    indices = np.where((y < -2.3) & (x < 0.1))
    return np.stack((neuronID[indices], x[indices], y[indices], z[indices]))


def chunky_window(a, window):
    xp = np.r_[a, np.nan + np.zeros((-len(a) % window,))]
    return xp.reshape(-1, window)


def correctPhotobleaching(raw, vps=6, error_bad_fit=False):
    """Apply photobleaching correction to raw signals, works on a whole heatmap of neurons

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019

    takes raw neural signals N (rows) neruons x T (cols) time points
    we require that each neuron is a row and that there are more time points than rows

    """

    # Make sure the data we are getting makes sense.
    assert (
        raw.ndim <= 2
    ), "raw fluorescent traces passed to correctPhotobleaching() have the wrong number of dimensions"

    if raw.ndim == 2:
        assert (
            raw.shape[1] > raw.shape[0]
        ), "raw fluorescent traces passed to correctPhotobleaching() have the wrong size! Perhaps it is transposed?"

    window_s = 12.6  # time of median filter window in seconds
    medfilt_window = np.int(
        np.ceil(window_s * vps / 2.0) * 2 + 1
    )  # make it an odd number of frames

    photoCorr = np.zeros_like(raw)  # initialize photoCorr

    performMedfilt = True

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
            popt[row, :], pcov[:, :, row], xVals = fitPhotobleaching(
                smoothed[row, :], vps
            )

            # If the exponentional we found fits more poorly than a flat line, just use a flat line.
            residual = raw[row, :] - expfunc(
                xVals, *popt[row, :]
            )  # its ok to have nan's here
            sum_of_squares_fit = np.nansum(np.square(residual))

            flatLine = np.nanmean(raw[row, :])
            residual_flat = raw[row, :] - flatLine
            sum_of_squares_flat = np.nansum(np.square(residual_flat))

            if sum_of_squares_fit > sum_of_squares_flat:
                # The fit does more poorly than a flat line
                # So don't do any photobleaching correction

                photoCorr[row, :] = np.copy(raw[row, :])
                num_FailsExpFit += 1
            else:
                photoCorr[row, :] = (
                    popt[row, 0] * raw[row, :] / expfunc(xVals, *popt[row, :])
                )

        if np.true_divide(num_FailsExpFit, N_neurons) > 0.5:
            print(
                (
                    "Uh oh!: The majority of neurons fail to exhibit exponential decay in their raw signals.\n"
                    + "this could be a sign of a bad recording. \n If this is the red channel its grounds for exclusion., "
                )
            )
    #          assert error_bad_fit is False, "The majority of neurons fail to exhibit exponential decay in their raw signal compared to flat line."

    elif raw.ndim == 1:
        smoothed = medfilt(raw, medfilt_window)
        popt, pcov, xVals = fitPhotobleaching(smoothed, vps)
        photoCorr = popt[0] * raw / expfunc(xVals, *popt)

    else:
        raise ValueError(
            "The  number dimensions of the data in correctPhotobleaching are not 1 or 2"
        )

    return photoCorr


def fitPhotobleaching(activityTrace, vps):
    """ "
    Fit an exponential.

    Accepts a single neuron's activity trace.

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019


    """
    assert (
        activityTrace.ndim == 1
    ), "fitPhotobleaching only works on single neuron traces"

    # Now we will fit an exponential, following along with the tuotiral here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    xVals = np.array(np.arange(activityTrace.shape[-1])) / np.float(vps)

    # Identify just the data that is not a NaN
    nonNaNs = np.logical_not(np.isnan(activityTrace))

    from scipy.optimize import curve_fit

    # set up some bounds on our exponential fitting parameter, y=a*exp(bx)+c
    num_recordingLengths = 8  # the timescale of exponential decay should not be more than eight recording lengths long
    # because otherwise we could have recorded for way longer!!

    # Scale the activity trace to somethign around one.
    scaleFactor = np.nanmean(activityTrace)
    activityTrace = activityTrace / scaleFactor

    bounds = (
        [0, 1 / (num_recordingLengths * np.nanmax(xVals)), 0],  # lower bounds
        [np.nanmax(activityTrace[nonNaNs]) * 1.5, 0.5, 2 * np.nanmean(activityTrace)],
    )  # upper bound

    # as a first guess, a is half the max, b is 1/(half the length of the recording), and c is the average
    popt_guess = [
        np.nanmax(activityTrace) / 2,
        2 / np.nanmax(xVals),
        np.nanmean(activityTrace),
    ]

    popt, pcov = curve_fit(
        expfunc, xVals[nonNaNs], activityTrace[nonNaNs], p0=popt_guess, bounds=bounds
    )

    ## Now we want to inspect our fit, find values that are clear outliers, and refit while excluding those
    residual = activityTrace - expfunc(xVals, *popt)  # its ok to have nan's here

    nSigmas = 3  # we want to exclude points that are three standard deviations away from the fit

    # Make a new mask that excludes the outliers
    excOutliers = np.copy(nonNaNs)
    with np.errstate(invalid="ignore"):
        excOutliers[(np.abs(residual) > (nSigmas * np.nanstd(residual)))] = False

    # Refit excluding the outliers, use the previous fit as initial guess
    # note we relax the bounds here a bit

    try:
        popt, pcov = curve_fit(
            expfunc,
            xVals[excOutliers],
            activityTrace[excOutliers],
            p0=popt,
            bounds=bounds,
        )
    except:
        popt, pcov = curve_fit(
            expfunc, xVals[excOutliers], activityTrace[excOutliers], p0=popt
        )

    # rescale the amplitude, a, back to full size
    popt[0] = scaleFactor * popt[0]

    # rescale the c parameter
    popt[2] = scaleFactor * popt[2]
    return popt, pcov, xVals


def expfunc(x, a, b, c):
    # type: (xVals, a, b, c) -> yVals
    return a * np.exp(-b * x) + c


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
    """Calculate curvature Kappa from the animal's centerline.
    This is a reimplementation of a snipeet of code from Leifer et al Nat Meth 2011
    https://github.com/samuellab/mindcontrol-analysis/blob/cfa4a82b45f0edf204854eed2c68fab337d3be05/preview_wYAML_v9.m#L1555
    Returns curvature in units of inverse worm lengths.

    More information on curvature generally: https://mathworld.wolfram.com/Curvature.html
    Kappa = 1 /R where R is the radius of curvature

    """
    numcurvpts = np.shape(centerlines)[1]
    diffVec = np.diff(centerlines, axis=1)
    # calculate tangential vectors
    atDiffVec = np.unwrap(np.arctan2(-diffVec[:, :, 1], diffVec[:, :, 0]))
    curv = np.unwrap(
        np.diff(atDiffVec, axis=1)
    )  # curvature kappa = derivative of angle with respect to path length
    # curv = kappa * L/numcurvpts
    curv = curv * numcurvpts  # To get cuvarture in units of 1/L
    return curv


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
    # Calculate mean curvature within an ROI.
    # Identify outliers (default 6 sigma)  and interpolate over them.
    curv_metric = np.mean(curv[:, ROI_start:ROI_end], axis=1)
    # Find outliers & interpolate over them
    outlier_ind = np.where(
        abs(curv_metric - np.mean(curv_metric)) > num_stds * np.std(curv_metric)
    )
    curv_metric[outlier_ind] = np.nan
    nans, x = nan_helper(curv_metric)
    curv_metric[nans] = np.interp(x(nans), x(~nans), curv_metric[~nans])
    return curv_metric
