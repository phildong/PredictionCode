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
    for condition in ['moving', 'chip', 'immobilized']:  # ['moving', 'immobilized', 'chip']:
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


import matplotlib.pyplot as plt



def calc_R2(y_true, y_pred):
    """calculate the coefficient of determination
    as defined here:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

    The goal is to have a transparent external way of calculating R^2
    seperate of regressions, as a final sanity check.
    """
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    R2 = (1 - u / v)
    return R2

def linfunc(x, m, b):
    return m * x + b

def linear_fit(activity, behavior):
    """Fit neural activity to behavior using a simple linear fit.
    """
    assert activity.ndim == 1, "linear fit funciton only works on a single neuron trace at a time"
    assert np.all(np.isfinite(activity)), "expects finite values"
    assert np.all(np.isfinite(behavior)), "expects finite values"

    #scale the behavior to something around 1 for ease of fitting
    #we will un-scale the results later
    beh_scalefactor = np.mean(behavior)
    beh = (1 / beh_scalefactor) * behavior

    # scale the activity to something around 1 for ease of fitting
    # we will un-scale the results later
    act_scalefactor = np.mean(activity)
    act = (1 / act_scalefactor) * activity

    from scipy.optimize import curve_fit

    bounds = ([-10, -15], # lower bound of m (slope) and  b offset
                [10, 15]) # upper bound of m(slope) and b offset

    popt_guess = [1, 0]  #as a first guess assume no slope and offset

    #y = mx + b where x is activity and y is behavior
    popt, pcov = curve_fit(linfunc, act, beh, p0=popt_guess, bounds=bounds)

    m = popt[0] * beh_scalefactor / act_scalefactor
    b = popt[1] * beh_scalefactor

    return m, b, pcov




# For each type of recording
for key in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML18_moving']:
    dset = data[key]['input']

    # For each recording
    for idn in dset.keys():

        for behavior, title in zip(['AngleVelocity', 'Eigenworm3'],  ['Velocity', 'Turn']):

            #Get the data
            moving = data[key]['input'][idn]
            valid_map = moving['Neurons']['I_valid_map']
            movingAnalysis = data[key]['analysis'][idn]

            splits = movingAnalysis['Training']

            indices_contig_trace_test = valid_map[splits[behavior]['Test']]
            train, test = splits[behavior]['Train'], splits[behavior]['Test']

            beh = moving['BehaviorFull'][behavior]
            time = moving['Neurons']['I_Time']

            beh_crop_noncontig = moving['Behavior_crop_noncontig'][behavior]
            time_crop_noncontig = moving['Neurons']['I_Time_crop_noncontig']

            splits = data[key]['analysis'][idn]['Training']
            train, test = splits[behavior]['Train'], splits[behavior]['Test']


            #Try to find the best fitting neuron here.
            activity = moving['Neurons']['I_smooth_interp_crop_noncontig']

            behPred = np.empty(moving['BehaviorFull'][behavior].shape)

            numNeurons = activity.shape[0]
            R2_local = np.empty(numNeurons)
            R2_local[:] = np.nan

            m = np.empty(numNeurons)
            b = np.empty(numNeurons)
            m[:] = np.nan
            b[:] = np.nan

            for neuron in np.arange(numNeurons):

                m[neuron], b[neuron], pcov = linear_fit(activity[neuron, [train]].flatten(), beh_crop_noncontig[train])



                behPred[:] = np.nan
                behPred[valid_map] = linfunc(activity[neuron, :], m[neuron], b[neuron])

                R2_local[neuron] = calc_R2(beh[valid_map][test], behPred[valid_map][test])


            bestNeuron = np.argmax(R2_local)
            behPred[valid_map] = linfunc(activity[bestNeuron, :], m[bestNeuron], b[bestNeuron])

            plt.figure(figsize=[12, 9])
            plt.plot(time[valid_map[0]:valid_map[-1]], beh[valid_map[0]:valid_map[-1]], label="measured")
            plt.plot(time[valid_map[0]:valid_map[-1]], behPred[valid_map[0]:valid_map[-1]], label="predicted")
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel(title + ' Behavior')

            plt.title(key + ' ' + idn +
                '\nPredicting ' + title + ' from neuron %i, m=%.2f, b=%.2f fit on training \n R2_local = %.2f on testset' % (
                bestNeuron, m[bestNeuron], b[bestNeuron], R2_local[bestNeuron]))

            import prediction.provenance as prov
            prov.stamp(plt.gca(),.55,.15)


print("Beginning to save best single neuron plots")
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("best_single_neurons.pdf")
for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
print("Saved best single neuron plots.")