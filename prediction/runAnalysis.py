from __future__ import division #give me floating point when I divide (standard in python3)

# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
import os
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
from prediction import userTracker
import SLM

###############################################    
# 
#    run parameters
#
###############################################

def actuallyRun(typ='AML32', condition = 'moving'):
#    typ  possible values AML32, AML18, AML70, AML175
#    condition possible values moving, immobilized, chip


    #typ = 'AML175' # possible values AML32, AML18, AML70, AML175
    #condition = 'moving' # Moving, immobilized, chip

    first = True # if 0true, create new HDF5 file
    transient = 0
    save = True
    ###############################################
    #
    #    load data into dictionary
    #
    ##############################################
    path = userTracker.dataPath()
    folder = os.path.join(path, '{}_{}/'.format(typ, condition))
    dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
    outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
    outLocData = os.path.join(path,'Analysis/{}_{}.hdf5'.format(typ, condition))


    #original data parameters
    dataPars = {'medianWindow':50,  # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':100,  # gauss window for angle velocity derivative. Acts on full (50Hz) data
            'rotate':False,  # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 6,  # gauss window for red and green channel
            'interpolateNans': 6,  #interpolate gaps smaller than this of nan values in calcium data
            'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
               }


# data parameters
    dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
            'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 3,  # gauss window for red and green channel
            'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
            'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
            }

    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
    keyList = np.sort(dataSets.keys())
    if save:
        dh.saveDictToHDF(outLocData, dataSets)

    ## results dictionary
    resultDict = {}
    for kindex, key in enumerate(keyList):
        resultDict[key] = {}
        resultDict[key]['pars'] = dataPars
    # analysis parameters

    pars ={'nCompPCA': 20, # no of PCA components (Andy wants to calculate all of them, not just 20,
                             # but for now we use 20 because 'None' fails when you save to the HD5 format
            'PCAtimewarp':False, #timewarp so behaviors are equally represented
            'trainingCut': 0.6, # what fraction of data to use for training
            'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
            'linReg': 'simple', # ordinary or ransac least squares
            'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
            'useRank': 0, # use the rank transformed version of neural data for all analyses
            'useDeconv': 0, # use the deconvolved transformed version of neural data for all analyses
            'useRaw': 0, # use the deconvolved transformed version of neural data for all analyses
            'nCluster': 10, # use the deconvolved transformed version of neural data for all analyses
            'useClust':False,# use clusters in the fitting procedure.
            'periods': np.arange(0, 300) # relevant periods in seconds for timescale estimate
             }



    behaviors = ['AngleVelocity', 'Eigenworm3', 'AngleAccel']

    ###############################################
    #
    # check which calculations to perform
    #
    ##############################################
    createIndicesTest = 1#True
    periodogram = 0
    half_period = 0
    hierclust = 0
    bta = 0
    pca = 1#False
    kato_pca = 0#False
    half_pca = 0
    corr = 1
    predNeur = 0
    predPCA = 0
    svm = 0
    lasso = 0
    elasticnet = 0
    slm = 0
    slm_shrub = 0
    lagregression = 0
    # this requires moving animals
    if condition != 'immobilized':
        predNeur = 0 #moniika had this at 1, but i'm setting it to zero
        svm = 0
        lasso = 0
        elasticnet = 1#True
        slm = 1
        slm_shrub = 0
        predPCA = 1
        lagregression = 0


    ###############################################
    #
    # create training and test set indices
    #
    ##############################################
    #Note we are going to do this all on the crop_noncontiguous datasets because
    # most of the time when we are comparing training and testing we need to be predicting
    # behavior from many neurons, and so we we have to use the interpolated dataset, but we still want
    # to omit NaNs that affect the majority of the neurons (like leading and trailing NaNs, or gaps, for example)
    if createIndicesTest:
        for kindex, key in enumerate(keyList):
            resultDict[key] = {'Training':{}}
            for label in behaviors:
                train, test = dr.createTrainingTestIndices(dataSets[key], pars, label=label)
                if transient:
                   train = np.where(dataSets[key]['Neurons']['I_Time_crop_noncontig']<4*60)[0]
                    # after 4:30 min
                   test = np.where((dataSets[key]['Neurons']['I_Time_crop_noncontig']>7*60)*(dataSets[key]['Neurons']['Time']<14*60))[0]
                   resultDict[key]['Training']['Half'] ={'Train':train}
                   resultDict[key]['Training']['Half']['Test'] = test
                else:
                     # add half split
                    midpoint = np.mean(dataSets[key]['Neurons']['I_Time_crop_noncontig'])
                    trainhalf = np.where(dataSets[key]['Neurons']['I_Time_crop_noncontig']<midpoint)[0]
                    testhalf = np.where(dataSets[key]['Neurons']['I_Time_crop_noncontig']>midpoint)[0]
                    resultDict[key]['Training']['Half'] = {'Train':trainhalf, 'Test': testhalf}
                resultDict[key]['Training'][label] = {'Train':train, 'Test': test}


        print "Done generating trainingsets"

    ###############################################
    #
    # calculate the periodogram of the neural signals
    #
    ##############################################
    if periodogram:
        print 'running periodogram(s)'
        raise RuntimeError("Periodogram routines were called, but these have not been updated to use the latest activity traces.")
        for kindex, key in enumerate(keyList):
            resultDict[key]['Period'] = dr.runPeriodogram(dataSets[key], pars, testset = None)
    # for half the sample each
    if half_period:
        print 'running half periodogram(s)'
        for kindex, key in enumerate(keyList):
            splits = resultDict[key]['Training']
            resultDict[key]['Period1Half'] = dr.runPeriodogram(dataSets[key], pars, testset = splits[behaviors[0]]['Train'])
            resultDict[key]['Period2Half'] = dr.runPeriodogram(dataSets[key], pars, testset = splits[behaviors[0]]['Test'])

    ###############################################
    #
    # correlation neurons and behavior
    #
    ##############################################
    if corr:
        print 'running Correlation.'
        for kindex, key in enumerate(keyList):
            resultDict[key]['Correlation'] = dr.behaviorCorrelations(dataSets[key], behaviors)
            #half1 =  resultDict[key]['Training'][behaviors[0]]['Train']
            #resultDict[key]['CorrelationHalf'] = dr.behaviorCorrelations(dataSets[key], behaviors, subset = half1)

    ###############################################
    #
    # run svm to predict discrete behaviors
    #
    ##############################################
    #TODO: Remove SVM entirely from codebase
    if svm:
        for kindex, key in enumerate(keyList):
            print 'running SVM.'
            raise RuntimeError("SVM was called. but this have not been updated to use the latest activity traces.")
            splits = resultDict[key]['Training']
            resultDict[key]['SVM'] = dr.discreteBehaviorPrediction(dataSets[key], pars, splits )

    ###############################################
    #
    # run PCA and store results
    #
    ##############################################
    #%%
    if pca:
        print 'running PCA'
        for kindex, key in enumerate(keyList):
            resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)
     #       resultDict[key]['PCARaw'] = dr.runPCANormal(dataSets[key], pars, useRaw=True)


            #correlate behavior and PCA
            #resultDict[key]['PCACorrelation']=dr.PCACorrelations(dataSets[key],resultDict[key], behaviors, flag = 'PCA', subset = None)
    ###############################################
    #
    # run Kato PCA
    #
    ##############################################
    #%%
    if kato_pca:
        raise RuntimeError("running Kato style PCA is no longer supported.")
        # TODO: LOW PRIORITY Remove Kato PCA entirely from codebase
        print 'running Kato et. al PCA'
        for kindex, key in enumerate(keyList):
            resultDict[key]['katoPCA'] = dr.runPCANormal(dataSets[key], pars, deriv = True)
            splits = resultDict[key]['Training']
            resultDict[key]['katoPCAHalf1'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = splits['Half']['Train'], deriv=True)

            resultDict[key]['katoPCAHalf2'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = splits['Half']['Test'], deriv=True)

    ###############################################
    #
    # run split first-second half PCA
    #
    ##############################################
    #%%
    if half_pca:
        raise RuntimeError("running half-split PCA is no longer supported.")
        # TODO: LOW PRIORITY Remove half-PCA entirely from codebase
        print 'half-split PCA'
        for kindex, key in enumerate(keyList):
            # run PCA on each half
            splits = resultDict[key]['Training']
            resultDict[key]['PCAHalf1'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = splits['Half']['Train'])
            resultDict[key]['PCAHalf2'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset =splits['Half']['Test'])
            resultDict[key]['PCArankCorr'] = dr.rankCorrPCA(resultDict[key])
    #%%
    ###############################################
    #
    # predict behavior from PCA
    #
    ##############################################
    if predPCA:
        for kindex, key in enumerate(keyList):
            print 'predicting behavior PCA'
            splits = resultDict[key]['Training']
            resultDict[key]['PCAPred'] = dr.predictBehaviorFromPCA(dataSets[key], \
                        splits, pars, behaviors)
    #%%
    ###############################################
    #
    # predict neural dynamics from behavior
    #
    ##############################################
    if predNeur:
        for kindex, key in enumerate(keyList):
            raise RuntimeError("predicting neural dynamics from beahvior is no longer supported.")
            print 'predicting neural dynamics from behavior'
            splits = resultDict[key]['Training']
            resultDict[key]['RevPred'] = dr.predictNeuralDynamicsfromBehavior(dataSets[key], splits, pars)
        plt.show()
    #%%
    ###############################################
    #
    # use agglomerative clustering to connect similar neurons
    #
    ##############################################
    if hierclust:
        #TODO: move this hierarchical clustering over to the loadData() functions and use it to set the order of the neurons
        for kindex, key in enumerate(keyList):
            print 'running clustering'
            resultDict[key]['clust'] = dr.runHierarchicalClustering(dataSets[key], pars)
    #%%
    ###############################################
    #
    # use behavior triggered averaging to create non-othogonal axes
    #
    ##############################################
    if bta:
        raise RuntimeError("BTA is no longer supported.")
        # TODO: LOW PRIORITY Remove BTA from codebase
        for kindex, key in enumerate(keyList):
            print 'running BTA'
            resultDict[key]['BTA'] =dr.runBehaviorTriggeredAverage(dataSets[key], pars)
    #%%
    ###############################################
    #
    # linear regression using LASSO
    #
    ##############################################
    if lasso:
        print "Performing LASSO.",
        for kindex, key in enumerate(keyList):

            splits = resultDict[key]['Training']
            resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, splits, plot=0, behaviors = behaviors)
            # calculate how much more neurons contribute
            tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key],splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
            for tmpKey in tmpDict.keys():
                resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])

            tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
            for tmpKey in tmpDict.keys():
                resultDict[key]['LASSO'][tmpKey]=tmpDict[tmpKey]

            # do converse calculation -- give it only the neurons non-zero in previous case
            subset = {}
            subset['AngleVelocity'] = np.where(np.abs(resultDict[key]['LASSO']['Eigenworm3']['weights'])>0)[0]
            subset['Eigenworm3'] = np.where(np.abs(resultDict[key]['LASSO']['AngleVelocity']['weights'])>0)[0]
            resultDict[key]['ConversePredictionLASSO'] = dr.runLinearModel(dataSets[key], resultDict[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'LASSO', subset = subset)


    #%%
    ###############################################
    #
    # linear regression using elastic Net
    #
    ##############################################
    if elasticnet:
        for kindex, key in enumerate(keyList):
            print 'Running Elastic Net',  key
            splits = resultDict[key]['Training']
            resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=0, behaviors = behaviors)
            # calculate how much more neurons contribute
            tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
            for tmpKey in tmpDict.keys():
                resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])

            tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'ElasticNet', behaviors = behaviors)
            for tmpKey in tmpDict.keys():
                resultDict[key]['ElasticNet'][tmpKey]=tmpDict[tmpKey]
            # do converse calculation -- give it only the neurons non-zero in previous case
            subset = {}
            subset['AngleVelocity'] = np.where(np.abs(resultDict[key]['ElasticNet']['Eigenworm3']['weights'])>0)[0]
            subset['Eigenworm3'] = np.where(np.abs(resultDict[key]['ElasticNet']['AngleVelocity']['weights'])>0)[0]
            resultDict[key]['ConversePredictionEN'] = dr.runLinearModel(dataSets[key], resultDict[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'ElasticNet', subset = subset)

            # run scrambled control
            #print 'Running Elastic Net scrambled'
            #resultDict[key]['ElasticNetRandomized'] = dr.runElasticNet(dataSets[key], pars,splits, plot=0, behaviors = behaviors, scramble=True)


    #%%
    ###############################################
    #
    # linear regression using SLM.py
    #
    ##############################################
    if slm:
        for kindex, key in enumerate(keyList):
            print 'Running SLM',  key

            splits = resultDict[key]['Training']
            
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['CMSVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            resultDict[key]['SLM'] = {'AngleVelocity': SLM.optimize_slm(time, neurons, velocity), 'Eigenworm3': SLM.optimize_slm(time, neurons, curvature)}

    #%%
    ###############################################
    #
    # linear regression using SLM.py with decision shrub
    #
    ##############################################
    if slm_shrub:
        for kindex, key in enumerate(keyList):
            print 'Running SLM_shrub',  key

            splits = resultDict[key]['Training']
            
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            resultDict[key]['SLM_shrub'] = {'AngleVelocity': SLM.optimize_slm(time, neurons, velocity, options={'decision_tree': True}), 'Eigenworm3': SLM.optimize_slm(time, neurons, curvature, options={'decision_tree': True})}



    #%%
    ###############################################
    #
    # lag-time fits of neural activity
    #
    ##############################################
    if lagregression:
        for kindex, key in enumerate(keyList):
            raise RuntimeError("lagregression not supported yet, because now we are tranistioning to datasets that are non contiguous so we need a mores sophisticated way of handling lags")
            print 'Running lag calculation',  key
            splits = resultDict[key]['Training']
            #resultDict[key]['LagLASSO'] = dr.timelagRegression(dataSets[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lags = np.arange(-18,19, 3))
            resultDict[key]['LagEN'] = dr.timelagRegression(dataSets[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lags = np.arange(-18,19, 3), flag='ElasticNet')

    #%%
    ###############################################
    #
    # save data as HDF5 file
    #
    ##############################################
    if save:
        dh.saveDictToHDF(outLoc, resultDict)
