import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import os

import dataHandler as dh
import userTracker

def en_optimize(error, fit, reg, X, Y, alphas, sigma = 14):
    # scalerX = StandardScaler()
    # scalerX.fit(X.T)
    # X_scaled = scalerX.transform(X.T).T

    # Yrs = Y.reshape(Y.size, 1)
    # scalerY = StandardScaler()
    # scalerY.fit(Yrs)
    # Y_scaled = scalerY.transform(Yrs).reshape(Y.size)

    X_train, X_cv = split_test(X, 0.2)
    Y_train, Y_cv = split_test(Y, 0.2)

    X_deriv = gaussian_filter(X, sigma = (0, sigma), order=1)
    Y_deriv = gaussian_filter(Y, sigma = sigma, order = 1)
    X_train_deriv = gaussian_filter(X_train, sigma = (0, sigma), order = 1)
    Y_train_deriv = gaussian_filter(Y_train, sigma = sigma, order = 1)

    best_error = np.inf
    best_alpha = alphas[0]
    if len(alphas) > 1:
        for alpha in alphas:
            result = minimize(error, np.zeros(X.shape[0]+1), args = (fit, reg, X_train, Y_train, X_train_deriv, Y_train_deriv, alpha))
            r2 = R2(result.x, fit, X_cv, Y_cv)
            err = result.fun
            print(alpha, r2, err)
            if err < best_error:
                best_error = error
                best_alpha = alpha
    
    result = minimize(error, np.zeros(X.shape[0]+1), args = (fit, reg, X, Y, X_deriv, Y_deriv, best_alpha))
    return result.x, best_alpha

    # unscaled_params = np.zeros(result.x.shape)
    # unscaled_params[1:] = np.sqrt(scalerY.var_/scalerX.var_)*result.x[1:]
    # unscaled_params[0] = result.x[0]*np.sqrt(scalerY.var_) - np.dot(np.sqrt(scalerY.var_/scalerX.var_)*scalerX.mean_, result.x[1:]) + scalerY.mean_
    
    # return unscaled_params, best_alpha

# target function
def best_neuron_f(X,P):
	Y = P[1]*X+P[0]
	return Y

def all_neurons_f(X, P):
    Y = np.dot(X.T, P[1:]) + P[0]
    return Y

def reg(X, P, l1_ratio = 0.9):
    #rms_norm = np.sqrt(np.sum(X*X, axis=1))/np.max(np.sqrt(np.sum(X*X, axis=1)))
    vec = P[1:]#/rms_norm
    return l1_ratio*np.sum(np.abs(vec)) + 0.5*(1-l1_ratio)*np.sum(vec*vec)

# error to be minimized
def error(P,f,reg,X,Y,X_deriv,Y_deriv,alpha):
    e = (1./(4*Y.size*np.var(Y)))*np.sum(np.power(Y-f(X,P),2))
    e += 10*(1/(4*Y.size*np.var(Y_deriv)))*np.sum(np.power(Y_deriv-f(X_deriv, P), 2))
    e += alpha*reg(X, P)
    
    return e

def R2(P, f, X, Y):
    return 1-np.sum(np.power(Y-f(X,P),2))/np.sum(np.power(Y-np.mean(Y),2))

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def best_neuron(neurons, velocity, test = 0.4, sigma = 14):
    neurons_train, neurons_test = split_test(neurons, test)
    velocity_train, velocity_test = split_test(velocity, test)
    n_neurons = neurons.shape[0]

    best_neuron_train_mse = np.dot(velocity_train,velocity_train)/velocity_train.shape[0]
    best_neuron_idx = -1
    best_neuron_coef = 0
    best_neuron_offset = 0
    for i in range(n_neurons):
        neuron = neurons_train[i,:]
        neuron_deriv = gaussian_filter(neuron, sigma = sigma)
        velocity_train_deriv = gaussian_filter(velocity_train, sigma = sigma)

        # returns parameters that minimize the error function
        result = minimize(error, np.zeros(2), args = (best_neuron_f, lambda x, p: 0, neuron, velocity_train, neuron_deriv, velocity_train_deriv, 0))
        opt_offset, opt_coef = result.x

        resid = velocity_train - opt_coef*neuron - opt_offset
        mse = np.dot(resid,resid)/velocity_train.shape[0]
        
        if mse < best_neuron_train_mse:
            best_neuron_train_mse = mse
            best_neuron_idx = i
            best_neuron_coef = opt_coef
            best_neuron_offset = opt_offset

    scorepred = R2([best_neuron_offset, best_neuron_coef], best_neuron_f, neurons_test[best_neuron_idx,:], velocity_test)
    score = R2([best_neuron_offset, best_neuron_coef], best_neuron_f, neurons_train[best_neuron_idx, :], velocity_train)
    
    return (best_neuron_idx, best_neuron_coef, best_neuron_offset, score, scorepred)

def elastic_net(neurons, velocity, test = 0.4):
    neurons_train, neurons_test = split_test(neurons, test)
    velocity_train, velocity_test = split_test(velocity, test)

    result, best_alpha = en_optimize(error, all_neurons_f, reg, neurons_train, velocity_train, np.logspace(-1.5, 1.5, 7))

    scorepred = R2(result, all_neurons_f, neurons_test, velocity_test)
    score = R2(result, all_neurons_f, neurons_train, velocity_train)

    return (result[1:], result[0], score, scorepred, best_alpha)

def main():
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_test_acc_penalty_1.pdf"))

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

            print("sigma = "+str(14*(time[1]-time[0])))

            sigma = 14
            neurons_deriv = gaussian_filter(neurons, sigma = (0, sigma), order=1)
            velocity_deriv = gaussian_filter(velocity, sigma = sigma, order = 1)
            curvature_deriv = gaussian_filter(curvature, sigma = sigma, order = 1)

            bn = best_neuron(neurons, velocity)
            en = elastic_net(neurons, velocity)
            _, test_idx = split_test(np.arange(velocity.size), 0.4)

            print("Best neuron: %d with coeff %0.3f and intercept %0.3f" % (bn[0], bn[1], bn[2]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn[3], bn[4]))
            print("Elastic net scores: %0.3f (train) and %0.3f (test) (alpha = %0.2g)" % (en[2], en[3], en[4]))

            fig, ax = plt.subplots(4,2,figsize=(25,20))

            ax[0,0].plot(en[0], 'b', label='ElasticNet')
            ax[0,0].plot(bn[0], bn[1], 'g.', markersize=10, label='Best Neuron')
            ax[0,0].set_title('Velocity Coefficients')
            ax[0,0].legend()

            ax[1,0].plot(time, velocity, 'k', lw=1)
            ax[1,0].plot(time, np.dot(neurons.T, en[0]) + en[1], 'b', lw=1)
            ax[1,0].set_title(r'Velocity SLM R_train = %0.3f, R_test = %0.3f ($\alpha = %0.2g$)' % (en[2], en[3], en[4]))
            ax[1,0].set_xlabel('Time (s)')
            ax[1,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            ax[2,0].plot(time, velocity_deriv, 'k', lw=1)
            ax[2,0].plot(time, np.dot(neurons_deriv.T, en[0]) + en[1], 'b', lw=1)
            ax[2,0].set_title(r'Acceleration SLM R_train = %0.3f, R_test = %0.3f ($\alpha = %0.2g$)' % (en[2], en[3], en[4]))
            ax[2,0].set_xlabel('Time (s)')
            ax[2,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity_deriv), np.max(velocity_deriv), facecolor='gray', alpha = 0.5)

            ax[3,0].plot(time, velocity, 'k', lw=1)
            ax[3,0].plot(time, neurons[bn[0],:]*bn[1]+bn[2], 'b', lw=1)
            ax[3,0].set_title('Velocity Best Neuron R_train = %0.3f, R_test = %0.3f' % (bn[3], bn[4]))
            ax[3,0].set_xlabel('Time (s)')
            ax[3,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            bn = best_neuron(neurons, curvature)
            en = elastic_net(neurons, curvature)
            _, test_idx = split_test(np.arange(curvature.size), 0.4)

            print("Best neuron: %d with coeff %0.3f and intercept %0.3f" % (bn[0], bn[1], bn[2]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn[3], bn[4]))
            print("Elastic net scores: %0.3f (train) and %0.3f (test) (alpha = %0.2g)" % (en[2], en[3], en[4]))

            ax[0,1].plot(en[0], 'b', label='ElasticNet')
            ax[0,1].plot(bn[0], bn[1], 'g.', markersize=10, label='Best Neuron')
            ax[0,1].set_title('Curvature Coefficients')
            ax[0,1].legend()

            ax[1,1].plot(time, curvature, 'k', lw=1)
            ax[1,1].plot(time, np.dot(neurons.T, en[0]) + en[1], 'b', lw=1)
            ax[1,1].set_title(r'Curvature SLM R_train = %0.3f, R_test = %0.3f ($\alpha = %0.2g$)' % (en[2], en[3], en[4]))
            ax[1,1].set_xlabel('Time (s)')
            ax[1,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            ax[2,1].plot(time, curvature_deriv, 'k', lw=1)
            ax[2,1].plot(time, np.dot(neurons_deriv.T, en[0]) + en[1], 'b', lw=1)
            ax[2,1].set_title(r'Acceleration SLM R_train = %0.3f, R_test = %0.3f ($\alpha = %0.2g$)' % (en[2], en[3], en[4]))
            ax[2,1].set_xlabel('Time (s)')
            ax[2,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature_deriv), np.max(curvature_deriv), facecolor='gray', alpha = 0.5)

            ax[3,1].plot(time, curvature, 'k', lw=1)
            ax[3,1].plot(time, neurons[bn[0],:]*bn[1]+bn[2], 'b', lw=1)
            ax[3,1].set_title('Curvature Best Neuron R_train = %0.3f, R_test = %0.3f' % (bn[3], bn[4]))
            ax[3,1].set_xlabel('Time (s)')
            ax[3,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            fig.suptitle(typ_cond+' '+key)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

    pdf.close()

if __name__ == '__main__':
    main()