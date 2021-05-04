import scipy.io
import os
import numpy as np
import matplotlib.pylab as plt

# set folder path

def loadCenterlines(folder, full=False, wormcentered=False):
    """get centerlines from centerline.mat file"""
    # cl = scipy.io.loadmat(folder+'centerline.mat')['centerline']
    tmp = scipy.io.loadmat(os.path.join(folder, 'centerline.mat'))

    cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder, 'centerline.mat'))['centerline'], 2, 0)
    if wormcentered:
        cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder, 'centerline.mat'))['wormcentered'], 1, 0)

    tmp = scipy.io.loadmat(os.path.join(folder, 'heatDataMS.mat'))

    clTime = np.squeeze(tmp['clTime'])  # 50Hz centerline times
    volTime = np.squeeze(tmp['hasPointsTime'])  # 6 vol/sec neuron times

    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime))))
    if not full:
        # reduce to volume time
        cl = cl[clIndices.astype(int)]
    # wcNew = wc[clIndices.astype(int)]
    # epNew = ep[clIndices.astype(int)]

    #    for cl in clNew[::10]:
    #        plt.plot(cl[:,0], cl[:,1])
    #    plt.show()
    return cl, clIndices.astype(int)




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

fold = '/projects/LEIFER/PanNeuronal/decoding_analysis/worm_data/AML32_moving/BrainScanner20200309_151024_MS'
clFull, clIndices = loadCenterlines(fold, full=True)
curv = getCurvature(clFull)

plt.figure()
plt.imshow(curv, vmin=-np.percentile(np.abs(curv),99.5), vmax=np.percentile(np.abs(curv),99.5), aspect='auto')
plt.show()