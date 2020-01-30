import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import os

import dataHandler as dh
import userTracker

# target function
def best_neuron_f(X,P):
	Y = P[1]*X+P[0]
	return Y

def all_neurons_f(X, P):
    Y = np.dot(X.T, P[1:]) + P[0]
    return Y

def reg(X, P, l1_ratio = 0.9):
    vec = P[1:]
    return l1_ratio*np.sum(np.abs(vec)) + 0.5*(1-l1_ratio)*np.sum(vec*vec)

# error to be minimized
def error(P,f,reg,X,Y,alpha):
    e = (1./(2*Y.size))*np.sum(np.power(Y-f(X,P),2))
    e += alpha*reg(X, P)
    
    return e

def R2(P, f, X, Y):
    return 1-np.sum(np.power(Y-f(X,P),2))/np.sum(np.power(Y,2))

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def best_neuron(neurons, velocity, test = 0.4):
    neurons_train, neurons_test = split_test(neurons, test)
    velocity_train, velocity_test = split_test(velocity, test)
    n_neurons = neurons.shape[0]

    best_neuron_train_mse = np.dot(velocity_train,velocity_train)/velocity_train.shape[0]
    best_neuron_idx = -1
    best_neuron_coef = 0
    best_neuron_offset = 0
    for i in range(n_neurons):
        neuron = neurons_train[i,:]

        # returns parameters that minimize the error function
        result = minimize(error, np.zeros(2), args = (best_neuron_f, lambda x, p: 0, neuron, velocity_train, 0))
        opt_offset, opt_coef = result.x

        resid = velocity_train - opt_coef*neuron - opt_offset
        mse = np.dot(resid,resid)/velocity_train.shape[0]
        
        if mse < best_neuron_train_mse:
            best_neuron_train_mse = mse
            best_neuron_idx = i
            best_neuron_coef = opt_coef
            best_neuron_offset = opt_offset

    velocity_predict = best_neuron_coef*neurons_test[best_neuron_idx,:]+best_neuron_offset
    resid = velocity_test-velocity_predict
    test_mse = np.dot(resid,resid)/resid.shape[0]

    score = 1 - best_neuron_train_mse*velocity_train.shape[0]/np.dot(velocity_train-np.mean(velocity_train), velocity_train-np.mean(velocity_train))
    scorepred = 1 - test_mse*velocity_test.shape[0]/np.dot(velocity_test-np.mean(velocity_test), velocity_test-np.mean(velocity_test))
    
    return (best_neuron_idx, best_neuron_coef, best_neuron_offset, score, scorepred)

def elastic_net(neurons, velocity, test = 0.4):
    neurons_train, neurons_test = split_test(neurons, test)
    velocity_train, velocity_test = split_test(velocity, test)

    alpha = 0.1

    result = minimize(error, np.zeros(neurons.shape[0]+1), args = (all_neurons_f, reg, neurons_train, velocity_train,alpha))

    scorepred = R2(result.x, all_neurons_f, neurons_test, velocity_test)
    score = R2(result.x, all_neurons_f, neurons_train, velocity_train)

    return (result.x[1:], result.x[0], score, scorepred)

def main():
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_test_en_reg.pdf"))

    for typ_cond in ['AML32_moving', 'AML70_chip', 'AML70_moving', 'AML18_moving']:
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
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            bn = best_neuron(neurons, velocity)
            en = elastic_net(neurons, velocity)
            _, test_idx = split_test(np.arange(velocity.size), 0.4)

            print("Best neuron: %d with coeff %0.3f and intercept %0.3f" % (bn[0], bn[1], bn[2]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn[3], bn[4]))
            print("Elastic net scores: %0.3f (train) and %0.3f (test)" % (en[2], en[3]))

            fig, ax = plt.subplots(3,2,figsize=(25,15))

            ax[0,0].plot(en[0], 'b', label='ElasticNet')
            ax[0,0].plot(bn[0], bn[1], 'g.', markersize=10, label='Best Neuron')
            ax[0,0].set_title('Velocity Coefficients')
            ax[0,0].legend()

            ax[1,0].plot(time, velocity, 'k', lw=1)
            ax[1,0].plot(time, np.dot(neurons.T, en[0]) + en[1], 'b', lw=1)
            ax[1,0].set_title('Velocity SLM R_train = %0.3f, R_test = %0.3f' % (en[2], en[3]))
            ax[1,0].set_xlabel('Time (s)')
            ax[1,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            ax[2,0].plot(time, velocity, 'k', lw=1)
            ax[2,0].plot(time, neurons[bn[0],:]*bn[1]+bn[2], 'b', lw=1)
            ax[2,0].set_title('Velocity Best Neuron R_train = %0.3f, R_test = %0.3f' % (bn[3], bn[4]))
            ax[2,0].set_xlabel('Time (s)')
            ax[2,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            bn = best_neuron(neurons, curvature)
            en = elastic_net(neurons, curvature)
            _, test_idx = split_test(np.arange(curvature.size), 0.4)

            print("Best neuron: %d with coeff %0.3f and intercept %0.3f" % (bn[0], bn[1], bn[2]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn[3], bn[4]))
            print("Elastic net scores: %0.3f (train) and %0.3f (test)" % (en[2], en[3]))

            ax[0,1].plot(en[0], 'b', label='ElasticNet')
            ax[0,1].plot(bn[0], bn[1], 'g.', markersize=10, label='Best Neuron')
            ax[0,1].set_title('Curvature Coefficients')
            ax[0,1].legend()

            ax[1,1].plot(time, curvature, 'k', lw=1)
            ax[1,1].plot(time, np.dot(neurons.T, en[0]) + en[1], 'b', lw=1)
            ax[1,1].set_title('Curvature SLM R_train = %0.3f, R_test = %0.3f' % (en[2], en[3]))
            ax[1,1].set_xlabel('Time (s)')
            ax[1,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            ax[2,1].plot(time, curvature, 'k', lw=1)
            ax[2,1].plot(time, neurons[bn[0],:]*bn[1]+bn[2], 'b', lw=1)
            ax[2,1].set_title('Curvature Best Neuron R_train = %0.3f, R_test = %0.3f' % (bn[3], bn[4]))
            ax[2,1].set_xlabel('Time (s)')
            ax[2,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            fig.suptitle(typ_cond+' '+key)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

    pdf.close()

if __name__ == '__main__':
    main()