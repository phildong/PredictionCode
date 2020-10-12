import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle

def rho_adj(y, yhat):
    truemean = np.mean(y)
    alpha = np.mean((yhat-truemean)*(y-yhat))

    truesigma = np.std(y)
    predsigma = np.std(yhat)
    rho2 = np.corrcoef(y, yhat)[0,1]**2

    return rho2 - (alpha)**2/((truesigma*predsigma)**2)

def rectified_derivative(neurons):
    nan_zero = np.copy(neurons)
    nan_zero[np.isnan(neurons)] = 0
    nan_zero_filtered = gaussian_filter(nan_zero, order = 1, sigma = (0, 7))

    flat = 0*neurons.copy()+1
    flat[np.isnan(neurons)] = 0
    flat_filtered = gaussian_filter(flat, order = 0, sigma = (0, 7))

    deriv = nan_zero_filtered/flat_filtered
    deriv_pos = np.copy(deriv)
    deriv_neg = np.copy(deriv)
    deriv_pos[deriv < 0] = 0
    deriv_neg[deriv > 0] = 0

    return deriv_pos, deriv_neg, deriv

pickled_data = 'comparison_results_curvature_l10.dat'
with open(pickled_data, 'rb') as handle:
    data = pickle.load(handle, encoding = 'bytes')

with open('neuron_data.dat', 'rb') as f:
    neuron_data = pickle.load(f, encoding = 'bytes')

res = data[b'BrainScanner20200130_110803'][b'slm_with_derivs']   

neurons_unn = neuron_data[b'BrainScanner20200130_110803'][b'neurons']
_, _, nderiv = rectified_derivative(neurons_unn)
neurons_and_derivs = np.vstack((neurons_unn, nderiv))

mean = np.mean(neurons_and_derivs, axis = 1)[:, np.newaxis]
std = np.std(neurons_and_derivs, axis = 1)[:, np.newaxis]
neurons = (neurons_and_derivs-mean)/std

fig = plt.figure(constrained_layout = True, figsize=(15, 10))
gs = fig.add_gridspec(3, 2)

performance = fig.add_subplot(gs[0,:])
neuron = fig.add_subplot(gs[2,:])
ts = fig.add_subplot(gs[1, 0])
sc = fig.add_subplot(gs[1, 1])

neuron.plot(res[b'time'], res[b'signal'], 'k', lw=1)
neur, = neuron.plot([], [], lw=2)
neuron.set_xlabel('Time (s)')
neuron.set_ylabel('Velocity')

ts.plot(res[b'time'], res[b'signal'], 'k', lw=1)
ts.set_xlabel('Time (s)')
ts.set_ylabel('Velocity')
ts.fill_between([res[b'time'][np.min(res[b'test_idx'])], res[b'time'][np.max(res[b'test_idx'])]], np.min(res[b'signal']), np.max(res[b'signal']), facecolor='gray', alpha = 0.5)
line, = ts.plot([], [], lw=2)

sc.plot([min(res[b'signal']), max(res[b'signal'])], [min(res[b'signal']), max(res[b'signal'])], 'k-.')
sc.set_xlabel('Measured Velocity')
sc.set_ylabel('Predicted Velocity')
trainpts, = sc.plot([], [], 'go', rasterized = True)
testpts, = sc.plot([], [], 'bo', rasterized = True)

order = np.argsort(-np.abs(res[b'weights']))

lines = np.zeros((order.size, res[b'time'].size))

for i in range(order.size):
    idxs = order[:i]
    output = res[b'intercepts'] + np.dot(res[b'weights'][idxs], neurons[idxs,:])

    coef = np.polyfit(output[res[b'train_idx']],res[b'signal'][res[b'train_idx']],1)
    poly1d_fn = np.poly1d(coef) 

    lines[i,:] = poly1d_fn(output)

rhos = np.array([rho_adj(res[b'signal'][res[b'test_idx']], lines[i,:][res[b'test_idx']]) for i in range(order.size)])

performance.set_ylim((0, 1))
performance.set_xlabel('Number of Neurons+Derivatives')
performance.set_ylabel(r'$\rho^2_\mathrm{adj}$')
performance.plot(range(rhos.size), rhos)

def init():
    line.set_data([], [])
    trainpts.set_data([], [])
    testpts.set_data([],[])
    neur.set_data([], [])
    fill_lines = performance.fill_between(range(0), np.zeros(0), rhos[:0], color='green', alpha = 0.5)
    return line, trainpts, testpts, neur, fill_lines

def animate(i):
    line.set_data(res[b'time'], lines[i,:])
    trainpts.set_data(res[b'signal'][res[b'train_idx']], lines[i,:][res[b'train_idx']])
    testpts.set_data(res[b'signal'][res[b'test_idx']], lines[i,:][res[b'test_idx']])
    neur.set_data(res[b'time'], neurons[i,:])
    fill_lines = performance.fill_between(range(i), np.zeros(i), rhos[:i], color='green', alpha=0.5)
    return line, trainpts, testpts, neur, fill_lines

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=order.size, interval=100, blit=True)
plt.show()
# anim.save('weights_animation_105620.mp4', writer='ffmpeg')