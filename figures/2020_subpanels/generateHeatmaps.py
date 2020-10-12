"""
Created Wed 8 January 2020
Gaol here is to generate useful output to understand how ro why changes to preprocessing
are affecting the performance of the different model predictions.
by Andrew Leifer
leifer@princeton.edu
"""

################################################
#
# grab all the data we will need
#
################################################
import os
import numpy as np

from prediction import userTracker
import prediction.dataHandler as dh






def main():

    codePath = userTracker.codePath()
    outputFolder = os.path.join(codePath,'figures/2020_subpanesl/generatedFigs')

    data = {}
    for typ in ['AKS297.51', 'AML32', 'AML18']:
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




    ### Plot Heatmap for each recording





    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)

    print("Plotting heatmaps.....")
    for key in ['AKS297.51_moving', 'AML32_moving', 'AML32_immobilized', 'AML18_moving', 'AML18_immobilized']:
        dset = data[key]['input']
        # For each recording
        for idn in dset.keys():

            dset = data[key]['input'][idn]


            #Get the relevant heatmaps
            I_smooth = dset['Neurons']['I_smooth']
            I_smooth_interp_crop_noncontig_wnans = np.copy(I_smooth)
            I_smooth_interp_crop_noncontig_wnans[:] = np.nan

            valid_map = dset['Neurons']['I_valid_map']
            I_smooth_interp_crop_noncontig_wnans[:, valid_map] = dset['Neurons']['I_smooth_interp_crop_noncontig']

            Iz = dset['Neurons']['ActivityFull']
            time = dset['Neurons']['I_Time']

            prcntile = 99.7
            fig = plt.figure(figsize=(18,20))
            plt.suptitle('data[' + key + '][' + idn + ']')


            ax = plt.subplot(4,1,1)
            pos = ax.imshow(I_smooth, aspect='auto',
                            interpolation='none', vmin=np.nanpercentile(I_smooth.flatten(), 0.1), vmax=np.nanpercentile(I_smooth.flatten(), prcntile),
                            extent = [ time[0], time[-1], 0, I_smooth.shape[0] ], origin='lower')
            ax.set_title('I_smooth_interp_noncontig  (smooth, common noise rejected, w/ NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)
            fig.tight_layout(rect=[0, 0.03, 1.1, 0.97])


            ax = plt.subplot(4,1,2)
            pos = ax.imshow(I_smooth_interp_crop_noncontig_wnans, aspect='auto',
                            interpolation='none', vmin=np.nanpercentile(I_smooth_interp_crop_noncontig_wnans,0.1), vmax=np.nanpercentile(I_smooth_interp_crop_noncontig_wnans.flatten(), prcntile),
                            extent = [ time[0], time[-1], 0, I_smooth_interp_crop_noncontig_wnans.shape[0] ], origin='lower')
            ax.set_title('I_smooth_interp_crop_noncontig_wnans  (smooth,  interpolated, common noise rejected, w/ large NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)


            ax = plt.subplot(4, 1, 3)
            pos = ax.imshow(Iz, aspect='auto',
                            interpolation='none', vmin=-2, vmax=2,
                            extent = [time[0], time[-1], 0, Iz.shape[0] ], origin='lower')
            ax.set_title('Activity  (per-neuron z-scored,  aggressive interpolation, common noise rejected,  Jeffs photobleach correction)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            fig.colorbar(pos, ax=ax)

            beh = dset['BehaviorFull']['AngleVelocity']
            time = dset['Neurons']['TimeFull']

            axbeh = plt.subplot(4, 1, 4)
            axbeh.plot(time, beh)
            fig.colorbar(pos, ax=axbeh)
            axbeh.axhline(linewidth=0.5, color='k')
            axbeh.set_ylim([-2, 4])
            axbeh.set_xlim(ax.get_xlim())

            axbeh.set_title('Velocity')
            axbeh.set_xlabel('Time (s)')
            axbeh.set_ylabel('Velocity (Body bends)')

            prov.stamp(plt.gca(),__file__,.9,.15)


    print("Beginning to save heat maps")
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(outputFolder, "heatmaps.pdf"))
    for fig in xrange(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print("Saved heatmaps.")





if __name__ == '__main__':
    print("about to run main()")
    main()