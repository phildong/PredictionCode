import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.backends.backend_pdf

import numpy as np
import numpy.ma as ma
import pickle
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, ifft

from utility import user_tracker
from utility import data_handler as dh

import os

from skimage.util.shape import view_as_windows as viewW
def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    # This function will roll each row of a matrix a, a an amount specified by r.
    # I got it here: https://stackoverflow.com/a/51613442/200688
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]

def nancorrcoef(A, B):
    a = ma.masked_invalid(A)
    b = ma.masked_invalid(B)

    msk = (~a.mask & ~b.mask)

    return ma.corrcoef(a[msk], b[msk])

def vcorrcoef(X,y):
    ''' vectorized corrcoef, from a https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/'''
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def get_pval_from_cdf(x,  rhos, cum_prob):
    np.all(np.diff(rhos) > 0)
    if x > 0:
        p = 1 - np.interp(x, rhos, cum_prob)
    if x <= 0:
        p = np.interp(x, rhos, cum_prob)
    min_pval=np.true_divide(1,cum_prob.shape[0]) # we can't claim probability is less than one over the number of shuffles
    return np.max([p, min_pval])

def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    # From https://stackoverflow.com/questions/39543002/returning-a-real-valued-phase-scrambled-timeseries
    fs = fft(ts)
    pow_fs = np.abs(fs) ** 2.
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    if len(ts) % 2 == 0:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)/2]
    else:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)/2 + 1]
    np.random.shuffle(phase_fsr_lh)
    if len(ts) % 2 == 0:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh,
                                    np.array((phase_fsr[len(phase_fsr)/2],)),
                                    phase_fsr_rh))
    else:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, phase_fsr_rh))
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsrp = ifft(fsrp)
    if not np.allclose(tsrp.imag, np.zeros(tsrp.shape)):
        max_imag = (np.abs(tsrp.imag)).max()
        print('\nNOTE: a non-negligible imaginary component was discarded.\n\tMax: {}' % max_imag)
    return tsrp.real

def shuffled_cdf_rho(activity, behavior, pdf=None, nShuffles=5000, shuffle_phase=False):
    '''Take recording of F and dF/dt for a set of N neurons, and shuffle
    each neuron nShuffles times. Calculate the Pearsons Correlation coefficient
    rho and get a distrubtion out.
    The distrubtion is the cumulative distribution of the rhos from the N x nShuffle
    '''
    assert(activity.shape[1] > 360), "The recording is less than 1 minute long, or the array is not in the expected format"
    import numpy.matlib
    print("Shuffling:", nShuffles * activity.shape[0])
    print("Time reversing and duplicating data...")
    shuff_activity = np.matlib.repmat(np.fliplr(activity), nShuffles, 1)
    assert (np.all(shuff_activity[4, :] == shuff_activity[4 + activity.shape[0], :])), "Somehow repmat failed"
    if shuffle_phase:
        print("Phase scrambling...")
        for k in np.arange(shuff_activity.shape[0]):
            shuff_activity[k,:] = phaseScrambleTS(shuff_activity[k,:])
    else:
        print("Generating Random Numbers...")
        roll = np.random.randint(activity.shape[1], size=nShuffles*activity.shape[0])
        assert (roll.shape[0] == shuff_activity.shape[0]), "The number of time lags does not match the number of rows of activity to shuffle"
        print("Permuting neural activity...")
        shuff_activity = strided_indexing_roll(shuff_activity, roll)
        assert (np.logical_not(np.all(np.all(shuff_activity[4,:] == shuff_activity[4+activity.shape[0],:])))), "By chance, two preselected rows were shuffled the same amount, or not at all. Should only happen p=1/(Number of samples in recording)"
    print("Calculating pearson's correlation coefficients...")
    rhos = vcorrcoef(shuff_activity, behavior)
    assert(rhos.shape[0]==shuff_activity.shape[0]), "Got the wrong number of corrcoefs rho"
    print("Finding CDF...")
    rhos = np.sort(rhos)
    cum_prob = np.linspace(0, 1, len(rhos), endpoint=False)
    print("Shuffled distribution found.")
    fig_cdf=plt.figure()
    plt.plot(rhos, cum_prob)
    plt.xlabel('rho')
    plt.ylabel('p')
    plt.title('CDF , N=%d, max(rho)=%.2f, min(rho)=%.2f' % (nShuffles*activity.shape[0], np.max(rhos), np.min(rhos)))
    if pdf is not None:
        pdf.savefig(fig_cdf)
    return rhos, cum_prob


def main(strain='AKS297.51_moving', recording='BrainScanner20200130_110803', behavior = 'velocity'):
    gtype = 'gfp' if '18' in strain else 'gcamp'

    with open('%s/%s_linear_models.dat' % (user_tracker.codePath(), gtype), 'rb') as f:
        data = pickle.load(f)
    with open('%s/%s_recordings.dat' % (user_tracker.codePath(), gtype), 'rb') as f:
        neuron_data = pickle.load(f)

    numShuffles = 500

    outfilename = recording + '_highweight_tuning_' + behavior + '.pdf'
    outputFolder = os.path.join(user_tracker.codePath(),'figures/output')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, outfilename))

    # Sort neurons by abs value weight
    w = data[recording][behavior][False]['weights']
    slm_weights_raw = w[:w.size / 2]
    slm_weights_raw_deriv = w[w.size / 2:]
    highly_weighted_neurons = np.flipud(np.argsort(np.abs(slm_weights_raw)))
    highly_weighted_neurons_deriv = np.flipud(np.argsort(np.abs(slm_weights_raw_deriv)))
    num_neurons=len(highly_weighted_neurons)

    # Calculate distribution of corrcoeff's on shuffled data for getting p-values
    activity_all = np.concatenate((neuron_data[recording]['neurons'], neuron_data[recording]['neuron_derivatives']), axis=0)
    rhos, cum_prob = shuffled_cdf_rho(activity_all, neuron_data[recording][behavior], pdf,nShuffles=numShuffles)

    # Make plot to populate with cumulative distribution function of rho
    fig_cdf = plt.figure()
    ax_cdf = fig_cdf.add_subplot(111)


    # Main Loop Through all neurons
    for ntype in ['F', 'dF/dt']:
        rho = np.zeros(num_neurons)
        pval = np.zeros(num_neurons)
        relevant_weighted_neurons = np.nan

        for rank in np.arange(num_neurons):
            if ntype == 'F':
                relevant_weighted_neurons = highly_weighted_neurons
                neuron = relevant_weighted_neurons[rank]
                weight = slm_weights_raw[neuron]
                activity = neuron_data[recording]['neurons'][neuron]
                color = u'#1f77b4'

            else:
                relevant_weighted_neurons = highly_weighted_neurons_deriv
                neuron = relevant_weighted_neurons[rank]
                weight = slm_weights_raw_deriv[neuron]
                activity = neuron_data[recording]['neuron_derivatives'][neuron]
                color = u'#ff7f0e'





            # Calculate bins for box plot and split data up into subarrays based on bin
            nbins = 11
            plus_epsilon = 1.00001
            bin_edges = np.linspace(np.nanmin(neuron_data[recording][behavior]) * plus_epsilon, np.nanmax(neuron_data[recording][behavior]) * plus_epsilon, nbins)
            binwidth = np.diff(bin_edges)
            assigned_bin = np.digitize(neuron_data[recording][behavior], bin_edges)
            activity_bin = [None] * (len(bin_edges) - 1)  # note the activity has to be lists, and there should be 1 less because the bins are edges
            for k, each in enumerate(np.unique(assigned_bin)):
                activity_bin[k] = activity[np.argwhere(assigned_bin == each)[:, 0]]

            rho[rank] = nancorrcoef(neuron_data[recording][behavior], activity)[0,1]
            pval[rank] = get_pval_from_cdf(rho[rank], rhos, cum_prob)
            fig1 = plt.figure(constrained_layout=True, figsize=[10, 5.3])
            gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig1)
            plt.rc('xtick', labelsize=17)
            plt.rc('ytick', labelsize=17)

            ax_blank=fig1.add_subplot(gs[1,0], title=recording + '  ' + ntype + '\n Neuron: %d,\n Weight Rank: %d, Weight = %.4f\n rho= %.2f, p=%.2E' % (neuron, rank, weight, rho[rank], pval[rank]))

            # Generate scatter plot and then box plot
            f1_ax1 = fig1.add_subplot(gs[0, 0], xlabel=behavior, ylabel='Activity (' + ntype + ')')
            f1_ax1.plot(neuron_data[recording][behavior], activity, 'o', alpha=.05, color=color)
            boxprops = dict(linewidth=.5)
            capprops = dict(linewidth=.5)
            whiskerprops = dict(linewidth=.5)
            flierprops = dict(linewidth=.2, markersize=1, marker='+')
            medianprops = dict(linewidth=2, color='k')#color='#67eb34')
            labels = [''] * len(activity_bin)
            try:
                f1_ax1.boxplot(activity_bin, positions=bin_edges[:-1] + binwidth / 2, widths=binwidth * .9, boxprops=boxprops,
                        medianprops=medianprops, labels=labels, manage_xticks=False,
                        capprops=capprops, whiskerprops=whiskerprops, flierprops=flierprops)
            except:
                print('Boxplot plotting errored and failed.')
            plt.locator_params(nbins=4)
            f1_ax1.axhline(linewidth=0.5, color='k')
            f1_ax1.tick_params(axis="x", labelsize=8)
            f1_ax1.tick_params(axis="y", labelsize=8)

            f1_ax2 = fig1.add_subplot(gs[0,1:], xlabel='time (s)', ylabel='Activity')
            f1_ax2.plot(neuron_data[recording]['time'], activity, color=color)
            f1_ax2.set_xlim(left=0)

            f1_ax3 = fig1.add_subplot(gs[1,1:], xlabel='time (s)', ylabel=behavior)
            f1_ax3.plot(neuron_data[recording]['time'], neuron_data[recording][behavior], color='black')
            f1_ax3.axhline(color='black')
            f1_ax3.set_xlim(left=0)
            pdf.savefig(fig1)

        # Plot summary of rho vs pvalue
        thresh=np.true_divide(0.05, 2 * num_neurons)
        sig_neg = np.sum((pval <= thresh) * (rho <= 0))
        sig_neg_minthresh = np.sum((pval <= thresh) * (rho < -0.4))
        sig_pos = np.sum((pval <= thresh) * (rho > 0))
        sig_pos_minthresh = np.sum((pval <= thresh) * (rho > 0.4))

        fig_sum = plt.figure(constrained_layout=True, figsize=[14, 5])
        ax = fig_sum.add_subplot(111)
        ax.plot(rho,pval,'o', color=color)
        ax.set_title(ntype + ' sig_neg=%d, sig_pos=%d\n also above thresh neg=%d, pos=%d' % (sig_neg, sig_pos, sig_neg_minthresh, sig_pos_minthresh) )
        ax.set_ylabel('p value')
        ax.set_xlabel(r'$\rho$')
        ax.set_yscale('log')
        ax.set_xlim(left=-.75,right=.75)
        ax.axhline(thresh, linestyle='--',color='red', linewidth=1)
        pdf.savefig(fig_sum)
        for i in np.arange(rho.shape[0]):
            ax.annotate(str(relevant_weighted_neurons[i]), (rho[i], pval[i] * (1+np.random.lognormal(.001))))
        pdf.savefig(fig_sum)
        fig_sum.clf()

        # Calculate and Plot CDF of rho
        cum_dist = np.argsort(np.abs(rho))
        ax_cdf.plot(np.abs(rho[cum_dist]), np.true_divide(np.arange(rho.shape[0]),rho.shape[0]), color=color, label=type)
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_xlabel(r'$|\rho|$')


    pdf.savefig(fig_cdf)
    pdf.close()
    print("wrote " + outfilename)

if __name__ == '__main__':
    main(behavior = 'velocity')
    main(behavior = 'curvature')

    # def find_high_weighted_neurons(strain, recording):
    #     print('Running tuning of highly weighted neurons for ', strain, recording)
    #     tuning_of_highly_weighted_neurons.main(strain=strain, recording='BrainScanner'+recording, behavior='velocity')
    #     tuning_of_highly_weighted_neurons.main(strain=strain, recording='BrainScanner'+recording, behavior='curvature')

    # strain = 'AKS297.51_moving'
    # find_high_weighted_neurons(strain, '20200130_110803') #AML_310_A
    # find_high_weighted_neurons(strain, '20200130_105254') #AML_310_B
    # find_high_weighted_neurons(strain, '20200310_142022') #AML_310_C
    # find_high_weighted_neurons(strain, '20200310_141211') #AML_310_D

    # strain = 'AML32_moving'
    # find_high_weighted_neurons(strain, '20170424_105620')  # AML32_A
    # find_high_weighted_neurons(strain, '20170610_105634')  # AML32_B
    # find_high_weighted_neurons(strain, '20170613_134800')  # AML32_C
    # find_high_weighted_neurons(strain, '20180709_100433')  # AML32_D
    # find_high_weighted_neurons(strain, '20200309_151024')  # AML32_E
    # find_high_weighted_neurons(strain, '20200309_153839')  # AML32_F
    # find_high_weighted_neurons(strain, '20200309_162140')  # AML32_G

    # strain = 'AML18_moving'
    # find_high_weighted_neurons(strain,'20200116_145254') #AML18_A
    # find_high_weighted_neurons(strain,'20200116_152636') #AML18_B
    # find_high_weighted_neurons(strain,'20200204_102136') #AML18_C
    # find_high_weighted_neurons(strain,'20200310_153952') #AML18_D
    # find_high_weighted_neurons(strain,'20200311_100140') #AML18_E
    # find_high_weighted_neurons(strain,'20200929_140030') #AML18_F
    # find_high_weighted_neurons(strain,'20200929_143439') #AML18_G
    # find_high_weighted_neurons(strain,'20210503_122703') #AML18_H
    # find_high_weighted_neurons(strain,'20210503_135244') #AML18_I
    # find_high_weighted_neurons(strain,'20210503_151831') #AML18_J
    # find_high_weighted_neurons(strain,'20210503_154404') #AML18_K