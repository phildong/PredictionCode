import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

import dataHandler as dh
import userTracker

def train_tree(X, Y):
    clf = tree.DecisionTreeClassifier(max_depth=1)
    samples = X.T
    vel_sign = np.sign(Y)
    clf = clf.fit(samples, vel_sign)
    return clf

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

    best_error = np.inf
    best_alpha = alphas[0]
    if len(alphas) > 1:
        for alpha in alphas:
            result = minimize(error, np.zeros(X.shape[0]+1), args = (fit, reg, X_train, Y_train, alpha))
            r2 = R2(result.x, fit, X_cv, Y_cv)
            err = result.fun
            print(alpha, r2, err)
            if err < best_error:
                best_error = error
                best_alpha = alpha
    
    result = minimize(error, np.zeros(X.shape[0]+1), args = (fit, reg, X, Y, best_alpha))
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
def error(P,f,reg,X,Y,alpha):
    e = (1./(4*Y.size*np.var(Y)))*np.sum(np.power(Y-f(X,P),2))
    e += alpha*reg(X, P)
    
    return e

def R2_raw(Y, Y_pred):
    return 1-np.sum(np.power(Y-Y_pred,2))/np.sum(np.power(Y-np.mean(Y),2))

def R2(P, f, X, Y):
    return 1-np.sum(np.power(Y-f(X,P),2))/np.sum(np.power(Y-np.mean(Y),2))

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def best_neuron(neurons_train, neurons_test, velocity_train, velocity_test, sigma = 14):
    n_neurons = neurons_train.shape[0]

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

    scorepred = R2([best_neuron_offset, best_neuron_coef], best_neuron_f, neurons_test[best_neuron_idx,:], velocity_test)
    score = R2([best_neuron_offset, best_neuron_coef], best_neuron_f, neurons_train[best_neuron_idx, :], velocity_train)
    
    return (best_neuron_idx, best_neuron_coef, best_neuron_offset, score, scorepred)

def elastic_net(neurons_train, neurons_test, velocity_train, velocity_test):
    result, best_alpha = en_optimize(error, all_neurons_f, reg, neurons_train, velocity_train, np.logspace(-1.5, 0.5, 5))

    scorepred = R2(result, all_neurons_f, neurons_test, velocity_test)
    score = R2(result, all_neurons_f, neurons_train, velocity_train)

    return (result[1:], result[0], score, scorepred, best_alpha)

def plot_multicolor(ax, x, y, idx, c1, c2):
    cmap = ListedColormap([c1, c2])
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(idx)

    ax.add_collection(lc)

def main():
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_test_split.pdf"))

    for typ_cond in ['AML32_moving']:#, 'AML70_chip', 'AML70_moving', 'AML18_moving']:
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

            neurons_train, neurons_test = split_test(neurons, test=0.4)
            velocity_train, velocity_test = split_test(velocity, test=0.4)
            curvature_train, curvature_test = split_test(curvature, test=0.4)

            clf = train_tree(neurons_train, velocity_train)
            print(clf.feature_importances_)
            vel_sign = clf.predict(neurons.T)
            vel_train_sign = clf.predict(neurons_train.T)
            vel_test_sign = clf.predict(neurons_test.T)

            train_idx, test_idx = split_test(np.arange(velocity.size), 0.4)

            pos_velocity_neurons = neurons[:,vel_sign > 0]
            neg_velocity_neurons = neurons[:,vel_sign < 0]

            pos_velocity_train = velocity_train[vel_train_sign > 0]
            pos_velocity_neurons_train = neurons_train[:,vel_train_sign > 0]
            pos_velocity_test = velocity_test[vel_test_sign > 0]
            pos_velocity_neurons_test = neurons_test[:,vel_test_sign > 0]
            neg_velocity_train = velocity_train[vel_train_sign < 0]
            neg_velocity_neurons_train = neurons_train[:,vel_train_sign < 0]
            neg_velocity_test = velocity_test[vel_test_sign < 0]
            neg_velocity_neurons_test = neurons_test[:,vel_test_sign < 0]

            bn_pos = best_neuron(pos_velocity_neurons_train, pos_velocity_neurons_test, pos_velocity_train, pos_velocity_test)
            bn_neg = best_neuron(neg_velocity_neurons_train, neg_velocity_neurons_test, neg_velocity_train, neg_velocity_test)
            en_pos = elastic_net(pos_velocity_neurons_train, pos_velocity_neurons_test, pos_velocity_train, pos_velocity_test)
            en_neg = elastic_net(neg_velocity_neurons_train, neg_velocity_neurons_test, neg_velocity_train, neg_velocity_test)

            velocity_pred_en = np.zeros(velocity.size)
            velocity_pred_en[vel_sign > 0] = np.dot(pos_velocity_neurons.T, en_pos[0]) + en_pos[1]
            velocity_pred_en[vel_sign < 0] = np.dot(neg_velocity_neurons.T, en_neg[0]) + en_neg[1]

            en_r2_train = R2_raw(velocity_train, velocity_pred_en[train_idx])
            en_r2_test = R2_raw(velocity_test, velocity_pred_en[test_idx])

            velocity_pred_bn = np.zeros(velocity.size)
            velocity_pred_bn[vel_sign > 0] = pos_velocity_neurons[bn_pos[0],:]*bn_pos[1] + bn_pos[2]
            velocity_pred_bn[vel_sign < 0] = neg_velocity_neurons[bn_neg[0],:]*bn_neg[1] + bn_neg[2]

            bn_r2_train = R2_raw(velocity_train, velocity_pred_bn[train_idx])
            bn_r2_test = R2_raw(velocity_test, velocity_pred_bn[test_idx])

            print("Best neurons: %d (forwards) and %d (backwards)" % (bn_pos[0], bn_neg[0]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn_r2_train, bn_r2_test))
            print("Elastic net scores: %0.3f (train) and %0.3f (test)" % (en_r2_train, en_r2_test))

            fig, ax = plt.subplots(3,2,figsize=(25,15))

            ax[0,0].plot(en_pos[0], color='blue', label='ElasticNet Positive')
            ax[0,0].plot(en_neg[0], color='red', label='ElasticNet Negative')
            ax[0,0].plot(bn_pos[0], bn_pos[1], 'b.', markersize=10, label='Best Neuron Positive')
            ax[0,0].plot(bn_neg[0], bn_neg[1], 'r.', markersize=10, label='Best Neuron Negative')
            ax[0,0].set_title('Velocity Coefficients')
            ax[0,0].legend()

            ax[1,0].plot(time, velocity, 'k', lw=1)
            ax[1,0].plot(time, velocity_pred_en, 'g', lw=1)
            ax[1,0].fill_between(time, min(velocity), max(velocity), where = vel_sign>0, color='blue', alpha=0.3)
            ax[1,0].fill_between(time, min(velocity), max(velocity), where = vel_sign<0, color='red', alpha=0.3)
            ax[1,0].set_title(r'Velocity SLM ($R^2_{\mathrm{test}} = %0.2g$, $R^2_{\mathrm{train}} = %0.2g$)' % (en_r2_test, en_r2_train))
            ax[1,0].set_xlabel('Time (s)')
            ax[1,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            ax[2,0].plot(time, velocity, 'k', lw=1)
            ax[2,0].plot(time, velocity_pred_bn, 'g', lw=1)
            ax[2,0].fill_between(time, min(velocity), max(velocity), where = vel_sign>0, color='blue', alpha=0.3)
            ax[2,0].fill_between(time, min(velocity), max(velocity), where = vel_sign<0, color='red', alpha=0.3)
            ax[2,0].set_title(r'Velocity Best Neuron ($R^2_{\mathrm{test}} = %0.2g$, $R^2_{\mathrm{train}} = %0.2g$)' % (bn_r2_test, bn_r2_train))
            ax[2,0].set_xlabel('Time (s)')
            ax[2,0].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(velocity), np.max(velocity), facecolor='gray', alpha = 0.5)

            clf = train_tree(neurons_train, curvature_train)
            print(clf.feature_importances_)
            curv_sign = clf.predict(neurons.T)
            curv_train_sign = clf.predict(neurons_train.T)
            curv_test_sign = clf.predict(neurons_test.T)

            pos_curvature_neurons = neurons[:,curv_sign > 0]
            neg_curvature_neurons = neurons[:,curv_sign < 0]

            pos_curvature_train = curvature_train[curv_train_sign > 0]
            pos_curvature_neurons_train = neurons_train[:,curv_train_sign > 0]
            pos_curvature_test = curvature_test[curv_test_sign > 0]
            pos_curvature_neurons_test = neurons_test[:,curv_test_sign > 0]
            neg_curvature_train = curvature_train[curv_train_sign < 0]
            neg_curvature_neurons_train = neurons_train[:,curv_train_sign < 0]
            neg_curvature_test = curvature_test[curv_test_sign < 0]
            neg_curvature_neurons_test = neurons_test[:,curv_test_sign < 0]

            bn_pos = best_neuron(pos_curvature_neurons_train, pos_curvature_neurons_test, pos_curvature_train, pos_curvature_test)
            bn_neg = best_neuron(neg_curvature_neurons_train, neg_curvature_neurons_test, neg_curvature_train, neg_curvature_test)
            en_pos = elastic_net(pos_curvature_neurons_train, pos_curvature_neurons_test, pos_curvature_train, pos_curvature_test)
            en_neg = elastic_net(neg_curvature_neurons_train, neg_curvature_neurons_test, neg_curvature_train, neg_curvature_test)

            curvature_pred_en = np.zeros(curvature.size)
            curvature_pred_en[curv_sign > 0] = np.dot(pos_curvature_neurons.T, en_pos[0]) + en_pos[1]
            curvature_pred_en[curv_sign < 0] = np.dot(neg_curvature_neurons.T, en_neg[0]) + en_neg[1]

            en_r2_train = R2_raw(curvature_train, curvature_pred_en[train_idx])
            en_r2_test = R2_raw(curvature_test, curvature_pred_en[test_idx])

            curvature_pred_bn = np.zeros(curvature.size)
            curvature_pred_bn[curv_sign > 0] = pos_curvature_neurons[bn_pos[0],:]*bn_pos[1] + bn_pos[2]
            curvature_pred_bn[curv_sign < 0] = neg_curvature_neurons[bn_neg[0],:]*bn_neg[1] + bn_neg[2]

            bn_r2_train = R2_raw(curvature_train, curvature_pred_bn[train_idx])
            bn_r2_test = R2_raw(curvature_test, curvature_pred_bn[test_idx])

            print("Best neurons: %d (forwards) and %d (backwards)" % (bn_pos[0], bn_neg[0]))
            print("Best neuron scores: %0.3f (train) and %0.3f (test)" % (bn_r2_train, bn_r2_test))
            print("Elastic net scores: %0.3f (train) and %0.3f (test)" % (en_r2_train, en_r2_test))

            ax[0,1].plot(en_pos[0], color='blue', label='ElasticNet Positive')
            ax[0,1].plot(en_neg[0], color='red', label='ElasticNet Negative')
            ax[0,1].plot(bn_pos[0], bn_pos[1], 'b.', markersize=10, label='Best Neuron Positive')
            ax[0,1].plot(bn_neg[0], bn_neg[1], 'r.', markersize=10, label='Best Neuron Negative')
            ax[0,1].set_title('Curvature Coefficients')
            ax[0,1].legend()

            ax[1,1].plot(time, curvature, 'k', lw=1)
            ax[1,1].plot(time, curvature_pred_en, 'g', lw=1)
            ax[1,1].fill_between(time, min(curvature), max(curvature), where = curv_sign>0, color='blue', alpha=0.3)
            ax[1,1].fill_between(time, min(curvature), max(curvature), where = curv_sign<0, color='red', alpha=0.3)
            ax[1,1].set_title(r'Curvature SLM ($R^2_{\mathrm{test}} = %0.2g$, $R^2_{\mathrm{train}} = %0.2g$)' % (en_r2_test, en_r2_train))
            ax[1,1].set_xlabel('Time (s)')
            ax[1,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            ax[2,1].plot(time, curvature, 'k', lw=1)
            ax[2,1].plot(time, curvature_pred_bn, 'g', lw=1)
            ax[2,1].fill_between(time, min(curvature), max(curvature), where = curv_sign>0, color='blue', alpha=0.3)
            ax[2,1].fill_between(time, min(curvature), max(curvature), where = curv_sign<0, color='red', alpha=0.3)
            ax[2,1].set_title(r'Curvature Best Neuron ($R^2_{\mathrm{test}} = %0.2g$, $R^2_{\mathrm{train}} = %0.2g$)' % (bn_r2_test, bn_r2_train))
            ax[2,1].set_xlabel('Time (s)')
            ax[2,1].fill_between([time[np.min(test_idx)], time[np.max(test_idx)]], np.min(curvature), np.max(curvature), facecolor='gray', alpha = 0.5)

            fig.suptitle(typ_cond+' '+key)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

    pdf.close()

if __name__ == '__main__':
    main()