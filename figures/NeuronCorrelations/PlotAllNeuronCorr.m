%Make the Neuron Correlation Plots, currently sorting them with cgIdxRev
%Does not seem to be the best clustering. 
%Sort both moving and immobile by same clustering currently- use immobile.
%Ideally this will sort/move all NaN to the end

%For data set 110803 (moving only)- frames 1-1465
%For data set 102246 (moving-immobile)-Moving 1-1590, Immobile 1700-4016 (maybe less/still immobile then)

%For data set 111126 (moving-immobile)-moving 1-1785, Immobile 1900-3360


cd '/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_test_new'
[NC_110803, cgIdxRev] = AllNeuron_Correlation(1, 1465);
%Change/display the neuron correlation by cgIdxRev?
neuronCor_110803 = NC_110803(cgIdxRev, cgIdxRev); %do I only sort x by cgIdx, or both?

figure
imagesc(neuronCor_110803,[-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-110803')
axis square


cd '/projects/LEIFER/PanNeuronal/20200313/BrainScanner20200313_102246_new_test'
[NC_102246_move, cgIdxRev_m] = AllNeuron_Correlation(1, 1590);
[NC_102246_im, cgIdxRev_i] = AllNeuron_Correlation(1700, 4016);
%Change/display the neuron correlation by cgIdxRev?
neuronCor_102246_move = NC_102246_move(cgIdxRev_i, cgIdxRev_i); %do I only sort x by cgIdx, or both?
neuronCor_102246_im = NC_102246_im(cgIdxRev_i, cgIdxRev_i);

figure
subplot(1, 3, 1)
imagesc(neuronCor_102246_move, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-102246-Move')
axis square

subplot(1, 3, 2)
imagesc(neuronCor_102246_im, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-102246-Immobile')
axis square

subplot(1, 3, 3)
imagesc(neuronCor_102246_move-neuronCor_102246_im, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-102246-Residuals')
axis square




cd '/projects/LEIFER/PanNeuronal/20200317/test_BrainScanner20200317_111126'
[NC_111126_move, cgIdxRev_m] = AllNeuron_Correlation(1, 1785);
[NC_111126_im, cgIdxRev_i] = AllNeuron_Correlation(1900, 3360);
%Change/display the neuron correlation by cgIdxRev?
neuronCor_111126_move = NC_111126_move(cgIdxRev_i, cgIdxRev_i); %do I only sort x by cgIdx, or both?
neuronCor_111126_im = NC_111126_im(cgIdxRev_i, cgIdxRev_i);

figure
subplot(1, 3, 1)
imagesc(neuronCor_111126_move, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-111126-Move')
axis square

subplot(1, 3, 2)
imagesc(neuronCor_111126_im, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-111126-Immobile')
axis square

subplot(1, 3, 3)
imagesc(neuronCor_111126_move-neuronCor_111126_im, [-1, 1])
xlabel('Sorted Neuron')
ylabel('Sorted Neuron')
title('Clustered Correlation-111126-Residuals')
axis square
