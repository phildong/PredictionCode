# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:34:10 2017
dimensionality reduction and linear model.
@author: monika
"""
import matplotlib.pylab as plt
import numpy as np
 
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn import svm
from sklearn.covariance import empirical_covariance
import dataHandler as dh
from scipy.interpolate import interp1d
from scipy.optimize import newton 
from sklearn.cluster import bicluster
from scipy.signal import welch, lombscargle, fftconvolve
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# 
import makePlots as mp

import userTracker

np.random.seed(13)
###############################################    
# 
# create trainings and test sets
#
############################################## 
def splitIntoSets(y, nBins=5, nSets=5, splitMethod='auto', verbose=0):
    """get balanced training sets for a dataset y."""
    # Determine limits of array
    yLimits = np.array([np.min(y),np.max(y)])
    #yLimits = np.percentile(y, [2.28, 97.72])#
    if verbose >= 1:
        print 'Min and max y: ', yLimits
        print 'Len(y): ', len(y)

    # Calculate bin edges and number of events in each bin
    nCounts,binEdges = np.histogram(y,nBins,yLimits)
    if  verbose >= 2:
        print 'Bin edges: ', binEdges
        print 'Counts: ', nCounts

    # Get minimum and maximum number of events in bins
    nCountsLimits = np.array([np.min(nCounts),np.max(nCounts)])
    if verbose >= 1:
        print 'Min and max counts: ', nCountsLimits

    # Determine bin index for each individual event
    # Digitize is semi-open, i.e. slightly increase upper limit
    binEdges[-1] += 1e-5
    binEdges[0], binEdges[-1] = np.min(y), np.max(y)+1e-5
    yBinIdx = np.digitize(y,binEdges) - 1
    
    # Get event indices for each bin
    eventIndices = []
    for binIdx in range(nBins):
        #print np.where(yBinIdx == binIdx).shape
        eventIndices.append(np.arange(len(y),dtype=int)[np.where(yBinIdx==binIdx)[0]])#[yBinIdx == binIdx])
    eventIndices = np.asarray(eventIndices)

    # Determine split method if auto is used
    nPerBin = nCountsLimits[0]/nSets
    if splitMethod == 'auto':
        if nPerBin < 10:
            splitMethod = 'redraw'
        else:
            splitMethod = 'unique'

    # Get proper number of events per bin, depending on split method
    if splitMethod == 'redraw':
        nPerBin = nCountsLimits[1]/nSets                            # Maximum bin count divided by number of sets
        if nPerBin > nCountsLimits[0]: nPerBin = nCountsLimits[0]   # But has to be at most the minimum bin count
    else: nPerBin = nCountsLimits[0]/nSets

    if verbose >= 1:
        print 'Split method: ', splitMethod
        print 'Events per bin per set: ', nPerBin

    # Create subsets
    sets = [[] for i in range(nSets)]
    for i in range(nBins):
        _tEvtIdx = np.asarray(eventIndices[i][:])
        for j in range(nSets):
            np.random.shuffle(_tEvtIdx)
            if len(_tEvtIdx) > nPerBin:
                sets[j].append(np.copy(_tEvtIdx[:nPerBin]))
            else:
                sets[j].append(np.copy(_tEvtIdx[:]))
            if splitMethod == 'unique':
                _tEvtIdx = _tEvtIdx[nPerBin:]

    # Convert into numpy arrays
    for i in range(nSets):
        sets[i] = np.asarray(sets[i]).reshape(nPerBin*nBins)
    sets = np.asarray(sets)

    # Prepare info dictionary with helpful data
    info = {}
    info['total-entries'] = int(sets.shape[0] * sets.shape[1])
    info['unique-entries'] = len(list(set(np.ravel(sets))))
    info['method'] = splitMethod
    info['min-max-y'] = yLimits
    info['min-max-cts'] = nCountsLimits
    info['nbins'] = nBins

    return sets, info
    
def createTrainingTestIndices(data, pars, label):
    """split time points into trainings and test set."""
    timeLen = data['Neurons']['I_smooth_interp_crop_noncontig'].shape[1]
    if pars['trainingType'] == 'start':
        cutoff = int(pars['trainingCut']*timeLen)
        testIndices = np.arange(timeLen)[:cutoff:]
        trainingsIndices = np.arange(timeLen)[cutoff::pars['trainingSample']]
    if pars['trainingType'] == 'simple':
        cutoff = int(pars['trainingCut']*timeLen)
        trainingsIndices = np.arange(timeLen)[:cutoff:pars['trainingSample']]
        testIndices = np.arange(timeLen)[cutoff:]
    elif pars['trainingType'] == 'random':
        cutoff = int(pars['trainingCut']*timeLen)
        tmpindices = np.arange(timeLen)
        np.random.shuffle(tmpindices[::pars['trainingSample']])
        trainingsIndices = np.sort(tmpindices[:cutoff])
        testIndices = np.sort(tmpindices[cutoff:])
    elif pars['trainingType'] == 'middle':
        cutoff = int((pars['trainingCut'])*timeLen/2.)
        testTime = int((1-pars['trainingCut'])*timeLen)
        tmpIndices = np.arange(timeLen)
#        if label =='Eigenworm3':
#            cutoff = int((1-pars['trainingCut'])*timeLen/2.)
#            loc = np.where(data['Behavior']['Ethogram']==2)[0]
#            loc = loc[np.argmin(np.abs(loc-timeLen/2.))]
#            testIndices = np.arange(np.max([0,loc-cutoff]), loc+cutoff)
#        else:
        # this makes a centered box
        testIndices = tmpIndices[cutoff:-cutoff]
        # this makes a box that starts in the center
        #testIndices = tmpIndices[int(timeLen/2.):int(timeLen/2.)+testTime]
        trainingsIndices = np.setdiff1d(tmpIndices, testIndices)[::pars['trainingSample']]
    elif pars['trainingType'] == 'LR':
        # crop out a testset first -- find an area that contains at least one turn
        #center = np.where(np.abs(data['Behavior']['Eigenworm3'])>15)[0]
        cutoff = int((pars['trainingCut'])*timeLen/2.)
        
        tmpIndices = np.arange(timeLen)
        testIndices = tmpIndices[cutoff:-cutoff]
        # create a trainingset by equalizing probabilities
        trainingsIndices = np.setdiff1d(tmpIndices, testIndices)
        # bin  to get probability distribution
        nbin = 5
        hist, bins = np.histogram(data['Behavior_crop_noncontig'][label], nbin)
        # this is the amount of data that will be left in each bin after equalization
        #N = np.sum(hist)/20.#hist[0]+hist[-1]
        N = np.max(hist)/2#*nbin
        print bins, np.min(data['Behavior_crop_noncontig'][label]), np.max(data['Behavior_crop_noncontig'][label])
        # digitize data 
        dataProb = np.digitize(data['Behavior_crop_noncontig'][label], bins=bins[:-2], right=True)
        # rescale such that we get desired trainingset length
        trainingsIndices= []
        
        tmpTime = np.arange(0,timeLen)
        
        np.random.shuffle(tmpTime)
        counter = np.zeros(hist.shape)
        for index in tmpTime:
                if index not in testIndices:
                    # enrich for rare stuff
                    n = dataProb[index]
                    if counter[n] <= N:
                        trainingsIndices.append(index)
                        counter[n] +=1
        print len(trainingsIndices)/1.0/timeLen, len(testIndices)/1.0/timeLen
        plt.hist(data['Behavior_crop_noncontig'][label], normed=True,bins=nbin )
        plt.hist(data['Behavior_crop_noncontig'][label][trainingsIndices], normed=True, alpha=0.5, bins=nbin)
        plt.show()
    return np.sort(trainingsIndices), np.sort(testIndices)
    


###############################################    
# 
# PCA
#
##############################################

def runPCANormal(data, pars, whichPC = 0, testset = None, deriv = False, useRaw=False):
    """run PCA on neural data and return nicely organized dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    if deriv:
        raise RuntimeError("Doing PCA on deriv is no longer supported.")
    if pars['useRank']:
        raise RuntimeError("Doing PCA on rankActivity is no longer supported")
    if useRaw:
        Neuro = data['Neurons']['I']
    else:
        Neuro = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig'])
    if testset is not None:
        Yfull = np.copy(Neuro).T
        Y = Neuro[:,testset].T
    else:
        Y= Neuro.T
        Yfull = Y
    # make sure data is centered
    sclar= StandardScaler(copy=True, with_mean=True, with_std=False)
    Y = sclar.fit_transform(Y)
    # neuron activity is transposed such that result = nsamples*nfeatures.
    comp = pca.fit_transform(Y).T
    pcs = pca.components_.T
    #print comp.shape, pcs.shape, 0/0
    if deriv:
        comp = np.cumsum(comp, axis=1)
    
    indices = np.argsort(pcs[:,whichPC])
    
    #print indices.shape
    pcares = {}
    pcares['nComp'] =  pars['nCompPCA']
    pcares['expVariance'] =  pca.explained_variance_ratio_
    pcares['eigenvalue'] =  pca.explained_variance_
    pcares['neuronWeights'] =  pcs
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  comp
    # reconstruct with nCompPCA
    sclar2= StandardScaler(copy=True, with_mean=True, with_std=False)
    Yfull = sclar2.fit_transform(Yfull)
    compFull = pca.transform(Yfull).T
    Yhat = np.dot(compFull.T[:,:nComp],pcs.T[:nComp,:])
    Yhat += sclar2.mean_
    if testset is not None:
        pcares['testSet'] = testset
    pcares['fullData'] = compFull
    pcares['reducedData'] = Yhat.T
    pcares['covariance'] = empirical_covariance(Yhat, assume_centered=False)
#    plt.subplot(311)        
#    plt.imshow(data['Neurons']['Activity'], aspect='auto')
#    plt.subplot(312)
#    plt.imshow(Yhat.T,  aspect='auto')
#    plt.subplot(313)
#    plt.imshow(pcares['covariance'])
#    plt.show()
    pcares['fullShuffle'], pcares['lagShuffle'] = runPCANoiseLevelEstimate(Y, pars)
    return pcares
###############################################    
# 
# noise level estimate PCA
#
##############################################

def runPCANoiseLevelEstimate(Y, pars):
    """run PCA on neural data and return nicely organized dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    # now we shuffle the data and calculate the variance explained
    YS = np.array([np.random.permutation(n) for n in np.copy(Y).T]).T
    # here we just randomly roll each timeseries by a certain amount i.e. lag or advance a neuron relative to others
    Nmax = Y.shape[0]
    YR = np.array([np.roll(n, np.random.randint(low=-Nmax, high=Nmax)) for n in Y.T]).T
    
    # make sure data is centered
    sclar= StandardScaler(copy=True, with_mean=True, with_std=False)
    YS = sclar.fit_transform(YS)
    pca.fit(YS)
    fullShuffle =  pca.explained_variance_ratio_
    # make sure data is centered
    sclar= StandardScaler(copy=True, with_mean=True, with_std=False)
    YR = sclar.fit_transform(YR)
    pca.fit(YR)
    lagShuffle =  pca.explained_variance_ratio_
#    plt.subplot(211)
#    plt.imshow(YS.T, aspect='auto')
#    plt.subplot(212)
#    plt.imshow(YR.T, aspect='auto')
#    plt.show()
    return fullShuffle, lagShuffle
    
def runPCATimeWarp(data, pars):
    """run PCA on neural data and return nicely orgainzed dictionary."""
    nComp = pars['nCompPCA']
    pca = PCA(n_components = nComp)
    neurons = timewarp(data)
    pcs = pca.fit_transform(neurons.T)
    
    pcs = pca.transform(np.copy(data['Neurons']['Activity']).T)
    
    # order neurons by weight in first component
    indices = np.arange(len( data['Neurons']['Activity']))
    indices = np.argsort(pca.components_[0])
    
    pcares = {}
    pcares['nComp'] =  pars['nCompPCA']
    pcares['expVariance'] =  pca.explained_variance_ratio_
    pcares['neuronWeights'] =  pca.components_.T
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  pcs.T
    
    return pcares

def timewarp(data):
    """creates a subsampled neuron signal for PCA that has equally represented behaviors."""
    # find out how much fwd and backward etc we have:
    #labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    neur = np.copy(data['Neurons']['Activity'])
    # find how often each behavior occurs
    indices = []
    for bindex, behavior in enumerate([-1,1,2]):
        # find behavior indices
        indices.append(np.where(data['Behavior']['Ethogram']==behavior)[0])
    # number of timestamps in each behavior
    lens = np.array([len(x) for x in indices])
    # timesamples in the smallest behavior
    minval = np.min(lens[np.nonzero(lens)])
    #subsample neural data to the minimal, non-zero size
    neurArr = []
    for i in range(3):
        
        if lens[i] !=0:
#            # this subsamples
#            useOnly = np.arange(0,lens[i], np.int(lens[i]/minval))[:minval]
#            tmp = neur[:,indices[i][useOnly]]
#            neurArr.append(tmp)
            # this averages
            tmp = neur[:,indices[i]]
            end =  minval * int(lens[i]/minval)
            neurArr.append(np.mean(tmp[:,:end].reshape(tmp.shape[0],-1, minval), 1))
            
    neurData = np.concatenate(neurArr, axis=1)
    return neurData

def rankCorrPCA(results):
    """correlate the first and second half PCA weights by rank."""
    tmpdata = np.zeros((3,3))
    for pc1 in range(3):
        for pc2 in range(3):
            # rank correlation
            #rankHalf1 = np.argsort(results['PCAHalf1']['neuronWeights'][:,pc1])
            #rankHalf2 = np.argsort(results['PCAHalf2']['neuronWeights'][:,pc2])
            #tmpdata[pc1, pc2] = np.corrcoef(rankHalf1, rankHalf2)[0,1]
            #dot product instead
            v1 = results['PCAHalf1']['neuronWeights'][:,pc1]
            v2 = results['PCAHalf2']['neuronWeights'][:,pc2]
            tmpdata[pc1, pc2] = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    return tmpdata
    
###############################################    
# 
# correlate neurons and behavior
#
##############################################
def behaviorCorrelations(data, behaviors, subset = None):
    """simple r2 scores of behavior and neural activity.
    updated Jan 2020 to handle NaNs and to use the
    mean and variance preserving activity traces
    """
    import numpy.ma as ma

    #Because each neuron is considered seperatley, we don't
    # need to perform any interpolation.. we just don't
    # include instances where there were NaNs

    Y = ma.masked_invalid(np.copy(data['Neurons']['I_smooth']))
    print Y.shape
    nNeur = Y.shape[0]
    results = {}
    
    for bindex, beh in enumerate(behaviors):
        r2s = []
        x = ma.masked_invalid(data['BehaviorFull'][beh])
        if subset is not None:
            raise RuntimeError("behaviorCorrelations() was called with a subset. But this feature hasn't been re-implemented yet.")
            print max(subset), Y.shape
            Y = Y[:,subset]
            x = x[subset]
        x = (x-ma.mean(x))/ma.std(x)
        for n in range(nNeur):
            r2s.append(ma.corrcoef(x, Y[n])[0,1]**2)
        results[beh] = np.array(r2s)
    return results
    
    
###############################################    
# 
# correlate neurons and PCA axes
#
##############################################
def PCACorrelations(data,results, behaviors, flag = 'PCA', subset = None):
    """simple r2 scores of behavior and neural activity."""
    Y = results[flag]['pcaComponents'][:3,]
    
    nNeur = Y.shape[0]
    results = {}
    
    for bindex, beh in enumerate(behaviors):
        r2s = []
        x = data['Behavior'][beh]
        if subset is not None:
            print max(subset), Y.shape
            Y = Y[:,subset]
            x = x[subset]
        x = (x-np.mean(x))/np.std(x)
        for n in range(nNeur):
            r2s.append(np.corrcoef(x, Y[n])[0,1]**2)
        results[beh] = np.array(r2s)
    return results
###############################################    
# 
# estimate signal/ periodogram
#
##############################################
###############################################
# 
# hierarchical clustering
#
##############################################    

def runHierarchicalClustering(data, pars, subset):
    """cluster neural data."""
    
    X = np.copy(data['Neurons']['Ratio']) # transpose to conform to nsamples*nfeatures
    if subset is not None:
        X = X[subset]
    # pairwise correlations
    C = np.ma.corrcoef(X)
    # find linkage
    Z = linkage(X, 'ward')
#    # assign clusters
    max_d = 1.5
#    clusters = fcluster(Z, max_d, criterion='distance')
    # assign clusters
    k=3
    clusters = fcluster(Z, k, criterion='maxclust')
    traces = []
    
    for index in np.unique(clusters):
        traces.append(X[np.where(clusters==index)])
    # store results
    clustres = {}
    clustres['linkage'] = Z
    clustres['clusters'] = traces
    clustres['leafs'] = clusters
    clustres['nclusters'] = len(np.unique(clusters))
    clustres['dmax'] = max_d
    clustres['threshold'] = Z[-(k-1),2]  

    return clustres

###############################################
# 
# Linear Model
#
##############################################    

def runLinearModel(data, results, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'LASSO', subset = None):
    """run a linear model to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = data['Behavior_crop_noncontig'][label]
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
    
        if pars['useRank']:
            raise RuntimeError("not supported")
            X = data['Neurons']['rankActivity'].T
        if pars['useClust']:
            raise RuntimeError("not supported")
            clustres = runHierarchicalClustering(data, pars)
            X = clustres['Activity'].T
        else:
            X = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig']).T # transpose to conform to nsamples*nfeatures
        if subset is not None:
            # only a few neurons
            if len(subset[label])<1:
                print 'no weights found.proceeding with all neurons'
            else:
                X = X[:,subset[label]]
#        cv = 10
        reg = linear_model.LinearRegression()
        # setting alpha to zero makes this a linear model without regularization
        reg = linear_model.Lasso(alpha = results[fitmethod][label]['alpha'])
        reg.fit(X[trainingsInd], Y[trainingsInd])
#        alphas = reg.alphas_
#        ymean = np.mean(reg.mse_path_, axis =1)
#        yerr = np.std(reg.mse_path_, axis =1)/np.sqrt(cv)
#        alphaNew = alphas[np.argmin(ymean)]
#        # calculate standard deviation rule
#        alphaNew = stdevRule(x = alphas, y= ymean, std= yerr)
#        reg = linear_model.Lasso(alpha=alphaNew)
#        reg.fit(X[trainingsInd], Y[trainingsInd])
        
        if plot:
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Y[trainingsInd], 'r')
            plt.plot(reg.predict(X[trainingsInd]), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(X[testInd]), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Y[testInd], reg.predict(X[testInd]), alpha=0.7, s=0.2)
#            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
#            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
#            ax1.set_xlabel(r'weights')
#            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
#            
#            #plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.3)
#            plt.plot(alphas, ymean, 'k')
#            plt.errorbar(alphas, ymean, yerr=yerr, capsize=1)
#            plt.axvline(alphas[np.argmin(ymean)],label='minimal error')
#            plt.axvline(alphaNew,label='stdev rule')
#            plt.xscale('log')
#            #plt.fill_between(reg.alphas_,y1=ymean-yerr, y2= ymean+yerr, alpha=0.5)
#            #plt.errorbar(,color= 'k')
            plt.tight_layout()            
            plt.show()
        weights = reg.coef_
        # score model
        if len(weights)>0:
            scorepred = reg.score(X[testInd], Y[testInd])#, sample_weight=np.power(Y[testInd], 2))
            score = reg.score(X[trainingsInd], Y[trainingsInd])
        else:
            scorepred = np.nan
            score = reg.score(X[trainingsInd], Y[trainingsInd])
        linData[label] = {}
        linData[label]['weights'] =  weights
        linData[label]['fullweights'] =  weights
        if subset[label] is not None and len(subset[label])>0:
            fullweights = np.zeros(data['Neurons']['I_smooth_interp_crop_noncontig'].shape[0])
            fullweights[subset[label]] = weights
            linData[label]['fullweights'] =  fullweights
        linData[label]['intercepts'] = reg.intercept_
#        linData[label]['alpha'] = alphaNew#reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        linData[label]['output'] = reg.predict(X) # full output training and test
        print 'r2', scorepred
    return linData


###############################################    
# 
# LASSO
#
##############################################   
def runLassoLars(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lag = None):
    """run LASSO to fit behavior and neural activity with a linear model."""
    raise RuntimeError("lassoLars() not updated yet.")

    linData = {}
    for label in behaviors:
        Y = np.reshape(np.copy(data['Behavior'][label]),(-1, 1))
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
    
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        elif pars['useClust']:
            clustres = runHierarchicalClustering(data, pars)
            X = clustres['Activity'].T
        else:
            X = np.copy(data['Neurons']['Activity'].T) # transpose to conform to nsamples*nfeatures
        # implement time lagging -- forward and reverse
        
        if lag is not None:
            # move neural activity by lag (lag has units of volume)
            # positive lag = behavior trails after neural activity, uses values from the past
            # negative lag = behavior precedes neural activity
            X = np.roll(X, shift = lag, axis = 0)
        # prep data by scalig
        # fit scale model
        scale = True
        if scale:
            scalerX = preprocessing.StandardScaler().fit(X[trainingsInd])  
            scalerY = preprocessing.StandardScaler().fit(Y[trainingsInd])  
            #scale data
            X = scalerX.transform(X)
            Y = scalerY.transform(Y)
        Xtrain, Xtest = X[trainingsInd],X[testInd]
        Ytrain, Ytest = Y[trainingsInd],Y[testInd]
        # fit lasso and validate
        #a = np.logspace(-2,2,100)
        cv = 10
        # unbalanced sets
#        fold = KFold(cv, shuffle=True) 
#        # balanced sets
        if label =='Eigenworm3':
            a = np.logspace(-3,-1,100)
        else:
            a = np.logspace(-3,0,100)
        if label =='Eigenworm3':
            fold = balancedFolds(Y[trainingsInd], nSets=cv)
##        else:
#        fold = balancedFolds(Y[trainingsInd], nSets=cv)
        fold = 5
        fold = TimeSeriesSplit(n_splits=5, max_train_size=None)
        reg = linear_model.LassoLarsIC(criterion = 'bic',  verbose=0, \
         max_iter=5000)#, eps=1e-2)#, normalize=False)
        
        reg.fit(Xtrain, Ytrain)
        alphas= reg.alphas_
        ymean = reg.criterion_
        yerr = None
        alphaNew = reg.alpha_
        ####for crossval
        #alphas = reg.cv_alphas_
        #ymean = np.mean(reg.mse_path_, axis =1)
        #yerr = np.std(reg.mse_path_, axis =1)/np.sqrt(cv)
        #alphaNew = alphas[np.argmin(ymean)]
#        # calculate standard deviation rule
#        alphaNew = stdevRule(x = alphas, y= ymean, std= yerr)
#        reg = linear_model.Lasso(alpha=alphaNew)
#        reg.fit(X[trainingsInd], Y[trainingsInd])
        
        if plot:
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Ytrain, 'r')
            plt.plot(reg.predict(Xtrain), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(Xtest), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Ytest, reg.predict(Xtest), alpha=0.7, s=0.2)
#            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
#            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
#            ax1.set_xlabel(r'weights')
#            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            
            #plt.plot(alphas, reg.mse_path_, 'k', alpha = 0.3)
            plt.plot(alphas, ymean, 'k')
            plt.errorbar(alphas, ymean, yerr=yerr, capsize=1)
            plt.axvline(alphas[np.argmin(ymean)],label='minimal error')
            #plt.axvline(alphaNew,label='stdev rule')
            plt.xscale('log')
            #plt.fill_between(reg.alphas_,y1=ymean-yerr, y2= ymean+yerr, alpha=0.5)
            #plt.errorbar(,color= 'k')
            plt.tight_layout()            
            plt.show(block=True)
        weights = reg.coef_
        # score model
        if len(weights)>0:
            # normalize testset with scaler values
            scorepred = reg.score(Xtest, Ytest)#, sample_weight=np.power(Y[testInd], 2))
            score = reg.score(Xtrain, Ytrain)
        else:
            scorepred = np.nan
            score = reg.score(Xtrain, Ytrain)
        linData[label] = {}
        linData[label]['weights'] =  weights
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = alphaNew#reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        if scale:
            #The output in normal behavior units is:
                # mean subtracted, normalized variance, NEURAL signals MULTIPLY BY the weights...
                # then undo the mean subtraction and normalized variance that was applied to the BEHAVIOR
            linData[label]['output'] = scalerY.inverse_transform(reg.predict(X)) # full output training and test
        else:
            linData[label]['output'] = reg.predict(X)
        print 'alpha', alphaNew, 'r2', scorepred
    return linData

###############################################    
# 
# LASSO
#
##############################################    

def stdevRule(x, y, std):
    """move by one stdeviation to increase regularization."""
    yUp = np.min(y) + std[np.argmin(y)]
    yFunc =  interp1d(x,y,'cubic',fill_value='extrapolate')
#    plt.plot(x,y)
#    plt.errorbar(x,y, yerr=std)
#    plt.show()
#    y0, y1 = np.min(y), y[-1]
#    x0, x1 = x[np.argmin(y)], x[-1]
#    m = (y1-y0)/(x1-x0)
#    c = y0-m*x0
#    xalpha = (c-yUp)/m
    xalpha = x[np.argmin(y)]*1.5
    xUpper = newton(lambda x: np.abs(yFunc(x) - yUp), xalpha)
#    print xUpper, xalpha
#    plt.plot(np.abs(yFunc(x) - yUp))
#    
#    plt.show()
    return xUpper
def balancedFolds(y, nSets=5,  splitMethod = 'unique'):
    """create balanced train/validate splitsby leave one out."""
    splits, _ = splitIntoSets(y, nBins=5, nSets=nSets, splitMethod=splitMethod, verbose=0)
    folds = []
    for i in range(len(splits)):
        folds.append([splits[i], np.concatenate(splits[np.arange(len(splits))!=i] )])
    return folds
    
def runLasso(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lag = None):
    """run LASSO to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = np.reshape(np.copy(data['Behavior_crop_noncontig'][label]),(-1, 1))
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
    
        if pars['useRank']:
            X = data['Neurons']['rankActivity'].T
        elif pars['useClust']:
            clustres = runHierarchicalClustering(data, pars)
            X = clustres['Activity'].T
        else:
            X = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig'].T) # transpose to conform to nsamples*nfeatures
        # implement time lagging -- forward and reverse
        
        if lag is not None:
            raise RuntimeError("Calculating time lags has not been updated to handle I_smooth_interp_crop_noncontig")
                #We can't just shift indices, because these are non-contiguous time series. There are gaps.
                # we would have to go back to the time index, move by one and then look up the corresponding volume
                # In any event, I don't have time to implement this now, so for now we just won't
                # use this functionality -- Andy 6 Jan 2020

            # move neural activity by lag (lag has units of volume)
            # positive lag = behavior trails after neural activity, uses values from the past
            # negative lag = behavior precedes neural activity
            X = np.roll(X, shift = lag, axis = 0)

        Xtrain, Xtest = X[trainingsInd],X[testInd]
        Ytrain, Ytest = Y[trainingsInd],Y[testInd]
        # fit lasso and validate
        #a = np.logspace(-2,2,100)
        #cv = 10
        # unbalanced sets
#        fold = KFold(cv, shuffle=True) 
#        # balanced sets
        if label =='Eigenworm3':
            a = np.logspace(-3,-1,100)
            nfold = 5#int(len(X)/250)
        else:
            a = np.logspace(-3,0,100)
            nfold =5#%int(len(X)/500)
        
        #if label =='Eigenworm3':
        #    nfold = balancedFolds(Y[trainingsInd], nSets=cv)
##        else:
#        fold = balancedFolds(Y[trainingsInd], nSets=cv)
        #fold = 5
        fold = TimeSeriesSplit(n_splits=nfold, max_train_size=None)
        reg = linear_model.LassoCV(cv=fold,  verbose=0, \
         max_iter=10000, tol=1e-4)#, alphas = a)#, normalize=False)
        
        reg.fit(Xtrain, Ytrain)
        alphas = reg.alphas_
        ymean = np.mean(reg.mse_path_, axis =1)
        yerr = np.std(reg.mse_path_, axis =1)/np.sqrt(nfold)
        alphaNew = alphas[np.argmin(ymean)]
#        # calculate standard deviation rule
#        alphaNew = stdevRule(x = alphas, y= ymean, std= yerr)
#        reg = linear_model.Lasso(alpha=alphaNew)
#        reg.fit(X[trainingsInd], Y[trainingsInd])
        
        if plot:
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Ytrain, 'r')
            plt.plot(reg.predict(Xtrain), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(Xtest), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Ytest, reg.predict(Xtest), alpha=0.7, s=0.2)
#            hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
#            ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
#            ax1.set_xlabel(r'weights')
#            ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            
            plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.3)
            plt.plot(alphas, ymean, 'k')
            plt.errorbar(alphas, ymean, yerr=yerr, capsize=1)
            plt.axvline(alphas[np.argmin(ymean)],label='minimal error')
            plt.axvline(alphaNew,label='stdev rule')
            plt.xscale('log')
            #plt.fill_between(reg.alphas_,y1=ymean-yerr, y2= ymean+yerr, alpha=0.5)
            #plt.errorbar(,color= 'k')
            plt.tight_layout()            
            plt.show(block=True)
        weights = reg.coef_
        # score model
        if len(weights)>0:
            # normalize testset with scaler values
            scorepred = reg.score(Xtest, Ytest)#, sample_weight=np.power(Y[testInd], 2))
            score = reg.score(Xtrain, Ytrain)
        else:
            scorepred = np.nan
            score = reg.score(Xtrain, Ytrain)
        linData[label] = {}
        linData[label]['weights'] =  weights
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = alphaNew#reg.alpha_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        if scale:
            linData[label]['output'] = scalerY.inverse_transform(reg.predict(X)) # full output training and test
        else:
            linData[label]['output'] = reg.predict(X)
        print 'alpha', alphaNew, 'r2', scorepred
    return linData
    
###############################################    
# 
# ElasticNet
#
##############################################    

def runElasticNet(data, pars, splits, plot = False, scramble = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lag = None):
    """run EN to fit behavior and neural activity with a linear model."""
    linData = {}
    for label in behaviors:
        Y = np.copy(data['Behavior_crop_noncontig'][label])
        Y = np.reshape(Y, (-1,1))
        #Y = preprocessing.scale(Y)
        if pars['useRank']:
            raise RuntimeError("using Rank w/ ElasticNet is not yet supported.")
            X = np.copy(data['Neurons']['rankActivity'].T)
        if pars['useRaw']:
            raise RuntimeError("using raw activity w/ ElasticNet is not yet supported.")
            X = np.copy(data['Neurons']['RawActivity'].T)
            X -= np.mean(X, axis = 0)
        elif pars['useClust']:
            raise RuntimeError("using Hierarchical Clustering w/ ElasticNet is not yet supported.")
            clustres = runHierarchicalClustering(data, pars)
            X = clustres['Activity'].T
        elif pars['useDeconv']:
            raise RuntimeError("using deconvolved activity w/ ElasticNet is not yet supported.")
            X = data['Neurons']['deconvolvedActivity'].T
        else:
            X = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig']).T # transpose to conform to nsamples*nfeatures
        if scramble:
            # similar to GFP control: scamble timeseries
            np.random.shuffle(Y)
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
        # implement time lagging -- forward and reverse
        if lag is not None:
            raise RuntimeError("Calculating effects of time lags not yet been updated to handle I_smooth_interp_crop_noncontig")
                #We can't just shift indices, because these are non-contiguous time series. There are gaps.
                # we would have to go back to the time index, move by one and then look up the corresponding volume
                # In any event, I don't have time to implement this now, so for now we just won't
                # use this functionality -- Andy 6 Jan 2020
            # move neural activity by lag (lag has units of volume)
            # positive lag = behavior lags after neural activity, uses values from the past
            X = np.roll(X, shift = lag, axis = 0)
        # fit scale model
        scale = 1
        if scale:
            scalerX = preprocessing.StandardScaler().fit(X[trainingsInd])  #neural activity
            scalerY = preprocessing.StandardScaler().fit(Y[trainingsInd])   #behavior
            #scale data
            X = scalerX.transform(X)
            Y = scalerY.transform(Y)
        Xtrain, Xtest = X[trainingsInd],X[testInd]
        Ytrain, Ytest = Y[trainingsInd],Y[testInd]
        # fit elasticNet and validate
        
        if label =='Eigenworm3':
            l1_ratio = [0.99]
            #l1_ratio = [0.95]
            #fold =10
            #fold = balancedFolds(Y[trainingsInd], nSets=cv)
            a = np.logspace(-2,-0.5,200)
            nfold = 5
            tol = 1e-10
        else:
            #l1_ratio = [0.5, 0.7, 0.8, .9, .95,.99, 1]
            l1_ratio = [0.99] #monika's value
            #fold = balancedFolds(Y[trainingsInd], nSets=cv)
            a = np.logspace(-4, -1, 200)

            nfold = 5
            tol = 1e-10
            
        #cv = 15
        #a = np.logspace(-3,-1,100)
       # fold = 5
        fold = TimeSeriesSplit(n_splits=nfold, max_train_size=None)
        reg = linear_model.ElasticNetCV(l1_ratio, cv=fold, verbose=0, selection='random', tol=tol, alphas=a)
        #        
        reg.fit(Xtrain, Ytrain)

        scorepred = reg.score(Xtest, Ytest)
        score = reg.score(Xtrain, Ytrain)
        
        #linData[label] = [reg.coef_, reg.intercept_, reg.alpha_, score, scorepred]
        linData[label] = {}
        linData[label]['weights'] = reg.coef_
        linData[label]['intercepts'] = reg.intercept_
        linData[label]['alpha'] = reg.alpha_
        linData[label]['l1_ratio'] = reg.l1_ratio_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred
        linData[label]['noNeurons'] = len(reg.coef_[np.abs(reg.coef_)>0])
        print 'R2', scorepred, 'N', len(reg.coef_[np.abs(reg.coef_)>0])
        if scale:
            #apply the inverse scaling for behavior.. and feed in the scaled activity
            linData[label]['output'] = scalerY.inverse_transform(reg.predict(X)) # full output training and test
        else:
            linData[label]['output'] = reg.predict(X)
        if plot:
            print 'alpha', reg.alpha_, 'l1_ratio', reg.l1_ratio_, 'r2', scorepred
            
            plt.subplot(221)
            plt.title('Trainingsset')
            plt.plot(Ytrain, 'r')
            plt.plot(reg.predict(Xtrain), 'k', alpha=0.7)
            plt.subplot(222)
            plt.title('Testset')
            plt.plot(Y[testInd], 'r')
            plt.plot(reg.predict(Xtest), 'k', alpha=0.7)
            ax1 = plt.subplot(223)
            plt.title('Non-zero weighths: {}'.format(len(reg.coef_[reg.coef_!=0])))
            ax1.scatter(Ytest, reg.predict(Xtest), alpha=0.7, s=0.2)
            #hist, bins = np.histogram(reg.coef_, bins = 30, density = True)
            #ax1.fill_between(bins[:-1],np.zeros(len(hist)), hist, step='post', color='r')
            #ax1.set_xlabel(r'weights')
            #ax1.set_ylabel('PDF(weights)')
            plt.subplot(224)
            # use if only one l1_ratio
            if len(l1_ratio)==1:
                plt.plot(reg.alphas_, reg.mse_path_, 'k', alpha = 0.1)
                plt.plot(reg.alphas_, np.mean(reg.mse_path_, axis =1))
            else:
                if len(reg.alphas_.shape)>1:
                    for l1index, l1 in enumerate(l1_ratio):
                        plt.plot(reg.alphas_[lindex], reg.mse_path_[l1index], 'k', alpha = 0.1)
                        plt.plot(reg.alphas_[lindex], np.mean(reg.mse_path_[l1index], axis =1))
                else:
                    for l1index, l1 in enumerate(l1_ratio):
                        plt.plot(reg.alphas_, reg.mse_path_[l1index], 'k', alpha = 0.1)
                        plt.plot(reg.alphas_, np.mean(reg.mse_path_[l1index], axis =1))
            plt.tight_layout()
            plt.show(block=True)
    return linData

###############################################    
# 
# run LASSO with multiple time lags and collate data
#
##############################################
    
def timelagRegression(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], flag = 'LASSO', lags = np.arange(-18,19, 3)):
    """runs LASSO in the same train/test split for multiple time lags and computes the standard erro, parameters etc."""
    # store results
    res = []
    for lag in lags:
        if flag =='LASSO':
            results = runLasso(data, pars, splits, plot = False, behaviors = behaviors, lag = lag)
        else:
            results = runElasticNet(data, pars, splits, plot = False, behaviors = behaviors, lag = lag)
        res.append(results)
    # pull out the results
    pcares = {}
    for rindex, results in enumerate(res):
        for lindex, label in enumerate(behaviors):
            if rindex == 0:
                pcares[label] = {}
                pcares[label]['lags'] = lags
            for key in results[label].keys():
                if rindex==0:
                    pcares[label][key] = []
                pcares[label][key].append(results[label][key])
                
    
    plt.figure()
    plt.subplot(221)
    for label in behaviors:
        plt.plot(pcares[label]['lags'], pcares[label]['scorepredicted'], label=label)
    plt.subplot(222)
    for label in behaviors:
        plt.plot(pcares[label]['lags'], pcares[label]['noNeurons'], label=label)
    plt.subplot(223)
    plt.imshow(np.array(pcares[behaviors[0]]['weights']), label=label, aspect='auto')
    plt.subplot(224)
    plt.imshow(np.array(pcares[behaviors[1]]['weights']), label=label, aspect='auto')
    plt.show()
    return pcares
    
###############################################    
# 
# run LASSO/EN with multiple train-test splits to obtain unbiased estimate of test error
# also check how robust model fitting is 
#
##############################################
    
def NestedRegression(data, pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], flag = 'LASSO'):
    """runs LASSO in the same train/test split for multiple time lags and computes the standard erro, parameters etc."""
    # store results
    res = []
    # since this is a time series, make sure test is after all training data
    timeLen = data['Neurons']['Activity'].shape[1]
    dur = pars['testVolumes'] # 2 min of data for test sets
    # make successive train-test splits with increasing time ('day-forward chaining')
    maxSplits = int((timeLen-dur)/dur)

    splitOuterLoop = [(np.arange(0, (i+1)*dur), np.arange((i+1)*dur, (i+2)*dur)) for i in range(maxSplits)]
    for repeats in splitOuterLoop:
        plt.plot(repeats[0])
        plt.plot(repeats[1])
        plt.show()
    splits = {}    
    
    for repeats in splitOuterLoop:
        for label in behaviors:
            splits[label] = {}
            splits[label]['Train'] , splits[label]['Test'] = repeats
        
        if flag =='LASSO':
            results = runLasso(data, pars, splits, plot = True, behaviors = behaviors, lag = None)
        else:
            results = runElasticNet(data, pars, splits, plot = False, behaviors = behaviors, lag = None)
        res.append(results)
    # pull out the results
    pcares = {}
    for rindex, results in enumerate(res):
        for lindex, label in enumerate(behaviors):
            if rindex == 0:
                pcares[label] = {}
                
            for key in results[label].keys():
                if rindex==0:
                    pcares[label][key] = []
                pcares[label][key].append(results[label][key])
                
    plt.figure()
    plt.subplot(221)
    for label in behaviors:
        plt.plot( pcares[label]['scorepredicted'], label=label)
    plt.subplot(222)
    for label in behaviors:
        plt.plot( pcares[label]['noNeurons'], label=label)
    plt.subplot(223)
    plt.imshow(np.array(pcares[behaviors[0]]['weights']), label=label, aspect='auto')
    plt.subplot(224)
    plt.imshow(np.array(pcares[behaviors[1]]['weights']), label=label, aspect='auto')
    plt.show()
    return pcares
    
###############################################    
# 
# calculate non-linearity from linear model output
#
##############################################
def fitNonlinearity(data, results, splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
    """fit a nonlinearity to a dataset that was fit with a linear model first."""
    for label in behaviors:
        # linear model output
        X = results[fitmethod][label]['output']
        # true values
        Y =  data['Behavior'][label]
        # 
        plt.hexbin(X,Y)
        plt.show()
###############################################    
# 
# Show how prediction improves with more neurons
#
##############################################  
def scoreModelProgression(data, results, splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
    """show how more neurons improve predictive abilities."""
    linData = {}
    for label in behaviors:
        # get the weights from previously fit data and sort by absolute amplitude
        weights = results[fitmethod][label]['weights']
        weightsInd = np.argsort(np.abs(weights))[::-1]
        
        # sort neurons by weight
        Y = data['Behavior_crop_noncontig'][label]
        if pars['useRank']:
            raise RuntimeError("using rank Activity is not supported.")
            X = data['Neurons']['rankActivity'].T
        else:
            X = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig']).T # transpose to conform to nsamples*nfeatures
        trainingsInd, testInd = splits[label]['Train'], splits[label]['Test']
        # individual predictive scores
        indScore = []
        sumScore = []
        mse = []
        print "___________________________"
        print fitmethod, 'params:', results[fitmethod][label]['alpha']
        print fitmethod, 'R2:', results[fitmethod][label]['scorepredicted'], results[fitmethod][label]['score']

        for count, wInd in enumerate(weightsInd):
            if np.abs(weights[wInd]) >0:
                # fit one neuron
                if fitmethod == 'LASSO':
                    reg = linear_model.Lasso(alpha = results[fitmethod][label]['alpha'])
                elif fitmethod == 'ElasticNet':
                    
                    reg = linear_model.ElasticNet(alpha = results[fitmethod][label]['alpha'],
                                                  l1_ratio = results[fitmethod][label]['l1_ratio'], tol=1e-5, selection='random')
                #reg = linear_model.LinearRegression()
                
                xTmp = np.reshape(X[:,weightsInd[:count+1]], (-1,count+1))
                reg.fit(xTmp[trainingsInd], Y[trainingsInd])
                
                sumScore.append(reg.score(xTmp[testInd], Y[testInd]))
                mse.append(np.sum((reg.predict(xTmp[testInd])-Y[testInd])**2))
                
                xTmp = np.reshape(X[:,wInd], (-1,1))
                reg.fit(xTmp[trainingsInd], Y[trainingsInd])
                indScore.append(reg.score(xTmp[testInd], Y[testInd]))
                
        linData[label] = {}
        linData[label]['cumulativeScore'] = sumScore
        #NOTE the indices of indivual score are NOT the natural indices..
        # on acocunt of the the if abs weight >0 statement
        # I need to completely redo the best single neuron analysis.

        linData[label]['individualScore'] = indScore
        linData[label]['MSE'] = mse
    return linData
    
###############################################    
# 
# reorganize linear models to fit PCA plot style
#
##############################################
def reorganizeLinModel(data, results, splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3']):
    """takes a model fit and calculates basis vectors etc."""
    # modify data to be like scikit likes it
    if pars['useRank']:
        raise RuntimeError("rank is not yet supported.")
        X = data['Neurons']['rankActivity'].T
    elif pars['useClust']:
        clustres = runHierarchicalClustering(data, pars)
        X = clustres['Activity']
    else:
        X = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig']).T # transpose to conform to nsamples*nfeatures
    # get weights
    pcs = np.vstack([results[fitmethod][label]['weights'] for label in behaviors])
    indices = np.argsort(pcs[0])
    
    comp = np.zeros((len(behaviors), len(X)))
    # show temporal components created by fits
    for wi, weights in enumerate(pcs):
        comp[wi] = np.dot(X, weights)
    # recreate neural data from weights
    neurPred = np.zeros(X.shape)
    
    # calculate predicted neural dynamics  TODO
    score = np.ones(len(behaviors))
    #score = explained_variance_score()
    
    pcares = {}
    pcares['nComp'] =  len(behaviors)
    pcares['expVariance'] =  score
    pcares['neuronWeights'] =  pcs.T
    pcares['neuronOrderPCA'] =  indices
    pcares['pcaComponents'] =  comp
    return pcares




###############################################    
# 
# predict behavior from 3 PCA axes
#
##############################################
def predictBehaviorFromPCA(data,  splits, pars, behaviors):
    linData = {}
    for label in behaviors:
        train, test = splits[label]['Train'], splits[label]['Test']

        behavior = data['Behavior_crop_noncontig'][label]
        # scale behavior to same 
        beh_scale = StandardScaler(copy=True, with_mean=True, with_std=True)
        behavior_zs = beh_scale.fit_transform(behavior.reshape(-1, 1)).flatten()
       
        # also reduce dimensionality of the neural dynamics.
        nComp = 3#pars['nCompPCA']
        pca = PCA(n_components = nComp)
        Neuro = np.copy(data['Neurons']['I_smooth_interp_crop_noncontig']).T

        # make sure data is centered
        sclar = StandardScaler(copy=True, with_mean=True, with_std=False)
        Neuro_mean_sub = sclar.fit_transform(Neuro)


        pcs = pca.fit_transform(Neuro_mean_sub)
        #now we use a linear model to train and test our predictions
        # lets build a linear model
        lin = linear_model.LinearRegression(normalize=False)
        lin.fit(pcs[train],  behavior_zs[train])
        
        score = lin.score(pcs[train],behavior_zs[train])
        scorepred = lin.score(pcs[test], behavior_zs[test])
        print 'PCA prediction results:'
        print 'Train R2: ',score 
        print 'Test R2: ', scorepred
        linData[label] = {}
        linData[label]['weights'] =  lin.coef_
        linData[label]['intercepts'] = lin.intercept_
        linData[label]['PCA_components'] = pca.components_
        linData[label]['score'] = score
        linData[label]['scorepredicted'] = scorepred

        # 1)take the mean-subtracted neural data, that is then PCA'd
        # 2) apply the weights
        # 3) un-z-score the behavioral result
        linData[label]['output'] = beh_scale.inverse_transform(lin.predict(pcs))
        
#        # check how well it performs with more PCs
#        r2_more = []
#        for nC in range(1, Neuro.shape[1], 1):
#            pca = PCA(n_components = nC)
#            pcs = pca.fit_transform(Neuro)
#            lin.fit(pcs[train], behavior[train])
#            r2_more.append(lin.score(pcs[test], behavior[test]))
#            
#        linData[label]['r2PCS'] = r2_more
#        linData[label]['r2PCSx'] = range(1, Neuro.shape[1], 1)
        
    return linData