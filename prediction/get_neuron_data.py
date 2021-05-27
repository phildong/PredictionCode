import SLM
from Classifier import rectified_derivative
import userTracker
import dataHandler as dh

import numpy as np
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter1d

import os

excludeSets = ['BrainScanner20200309_154704', 'BrainScanner20181129_120339', 'BrainScanner20200130_103008']
excludeInterval = {'BrainScanner20200309_145927': [[50, 60], [215, 225]], 
                   'BrainScanner20200309_151024': [[125, 135], [30, 40]], 
                   'BrainScanner20200309_153839': [[35, 45], [160, 170]], 
                   'BrainScanner20200309_162140': [[300, 310], [0, 10]],
                   'BrainScanner20200130_105254': [[65, 75]],
                   'BrainScanner20200310_141211': [[200, 210], [240, 250]]}

def calc_head_angle(centerlines):
    nose_start = 0
    nose_to_neck = 10
    neck_end = 20

    nose_vec = np.diff(centerlines[:, [nose_start, nose_to_neck], :], axis=1)
    neck_vec = np.diff(centerlines[:, [nose_to_neck, neck_end], :], axis=1)

    nose_orientation = np.zeros(centerlines.shape[0])
    neck_orientation = np.zeros(centerlines.shape[0])

    for timept in np.arange(centerlines.shape[0]):
        nose_orientation[timept] = np.arctan2(nose_vec[timept,:,0], nose_vec[timept,:,1])
        neck_orientation[timept] = np.arctan2(neck_vec[timept,:,0], neck_vec[timept,:,1])

    head_angle = nose_orientation - neck_orientation

    #We want to do the opposite of the unwrap function, such that everything is on the 0 to 2*pi scale, as described here:
    # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap/15927914
    head_angle = (head_angle + np.pi) % (2 * np.pi) - np.pi

    return head_angle

def find_phase_4pt(pospeaks, negpeaks, zero_crossings, time):
    '''Find the phase  by taking into account  positive and negative peaks and zero crosisngs.
    '''
    #Get arrays that have the time points (although they aren't ordered)
    keypt_phase_indices = np.concatenate((pospeaks, negpeaks), axis=0)
    ind = np.argsort(keypt_phase_indices) #sort them by index
    keypt_phase_indices =  keypt_phase_indices[ind]

    #Now we want to get the up and down peaks onto a single timeline, so that we can detect weird cases where there are \
    # two peaks and no trough in between

    # Define an array that specifies whether the peak is positive or negative
    updown = np.concatenate((np.ones(pospeaks.size), np.zeros(negpeaks.size)),
                            axis=0)
    updown = updown[ind] # sort them also by time


    # First lets get the phase going and allow it to go linearly beyond 2pi. we'll take the modulo later.
    cum_phase = np.cumsum(np.abs(np.diff(updown))) * np.pi
                                        # (whats going on here is that the diff will be +/-1 everytime
                                        # there is a  peak to trough or trough to peak transition
                                        # it will be zero if there are two peaks or two troughs in a row
                                        # so if we take the cumsum and multiply by pi we get the phases
                                        # that we can interpolate over later)


    # Let's deal with some additioanl subtelties.
    # We tack on a zero because we skip one point when we take the diff
    cum_phase = np.append(0, cum_phase)


    #Then we want to set the the first positive peak to have phase of zero
    # so we subtract off phase value at the first positive peak  (depending on whether a positive or negative peak came first,
    # this could have been assigned a pi value)
    cum_phase = cum_phase - cum_phase[np.where(updown == 1)[0][0]]

    # Then we need to deal with the fact that the recording doesn't necessarily start with a peak,
    # So we will be extrapolating whatever trend back towards zero as we go back in time from the first peak
    # We would prefer that the phase never goes negative, so we will add some additional factors of 2 pi
    cum_phase = cum_phase + 4 * np.pi
    # this is ok because we take the modulo later



    # Now we interpolate and extroplate to assign a phase value to every point in time
    from scipy import interpolate
    f = interpolate.interp1d(time[keypt_phase_indices], cum_phase, fill_value="extrapolate")
    cum_phase_interp = f(time)


    # ANDYS IDEA FOR ZERO CROSSINGS: to get the zero crossings, what we want to do is force the phase to be  
    # pi/2 or 3pi / 2 at each zero crossings.
    # we can acheive this by simply rounding to the nearest of those two options. Then we would re-interpolate


    # To force roundign to only pi/2 or 3pi/2 here is a little recipe:
    #divide by pi  (one cycle runs 0 to 2)
    #then add 1/2 (one cycle now runs .5 to 2.5)
    #round (lock to 1 or 2)
    #then subtract 1/2
    #then multiply by pi again
    cum_phase_zero_cross =  ( np.around( np.true_divide(cum_phase_interp[zero_crossings], np.pi) + 0.5 ) - 0.5 ) * np.pi

    # Now combine the zero corssing phases  with the peak and trough phases, and do the same for the time indices
    cum_phase2 = np.concatenate((cum_phase, cum_phase_zero_cross), axis=0)
    keypt_phase_indices = np.concatenate((keypt_phase_indices, zero_crossings), axis=0)
    
    ind = np.argsort(keypt_phase_indices) #sort them by index
    cum_phase = cum_phase2[ind]

    # Now repeat the interpolation
    f2 = interpolate.interp1d(time[keypt_phase_indices[ind]], cum_phase, fill_value="extrapolate")
    cum_phase_interp = f2(time)

    # now chop it into range 0 to 2pi
    phase = np.mod(cum_phase_interp, 2*np.pi)
    return phase

data = {}
for typ_cond in ['AKS297.51_moving', 'AML32_moving']:
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
        if key in excludeSets:
            continue
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
        cmsvelocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
        curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']
   

        phasevelocity = dataSets[key]['Behavior_crop_noncontig']['PhaseShiftVelocity']
        grosscurvature = dataSets[key]['Behavior_crop_noncontig']['Curvature']

        centerlines = np.array(dataSets[key]['CLFull'])
        timefull = np.array(dataSets[key]['Neurons']['TimeFull'])

        head_angle = np.unwrap(calc_head_angle(centerlines))
        sigma = 3
        smooth_head_angle = dh.gauss_filterNaN(head_angle, sigma)

        from scipy.signal import find_peaks
        prominence = 0.4
        peaks, _ = find_peaks(smooth_head_angle, height=-.3, prominence=prominence)
        neg_peaks, _ = find_peaks(smooth_head_angle*-1, height=-.3, prominence=prominence)

        zero_crossings = np.where(np.diff(np.sign(smooth_head_angle)))[0]

        phase = find_phase_4pt(peaks, neg_peaks, zero_crossings, timefull)

        phase_interp = np.interp(time, timefull, phase)

        if key in excludeInterval.keys():
            for interval in excludeInterval[key]:
                idxs = np.where(np.logical_or(time < interval[0], time > interval[1]))[0]
                time = time[idxs]
                neurons = neurons[:,idxs]
                velocity = velocity[idxs]
                cmsvelocity = cmsvelocity[idxs]
                curvature = curvature[idxs]
                phase_interp = phase_interp[idxs]
                phasevelocity = phasevelocity[idxs]
                grosscurvature = grosscurvature[idxs]

        data[key] = {'time': time, 'neurons': neurons, 'velocity': velocity, 'cmsvelocity': cmsvelocity, 'curvature': curvature, 'head_bend': phase_interp, 'phase_velocity': phasevelocity, 'gross_curvature': grosscurvature}

import pickle
with open('neuron_data_bothmc_nb.dat', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
