function [neuronCorrelation, cgIdxRev] = AllNeuron_Correlation(startFrame, endFrame)


%For data set 110803 (moving only)- frames 1-1465
%For data set 102246 (moving-immobile)-Moving 1-1590, Immobile 1700-4016 (maybe less/still immobile then)

%For data set 111126 (moving-immobile)-moving 1-1785, Immobile 1900-3360


%go through and find the correlation of each neuron to each neuron 
load('heatData.mat')

%for each neuron do the correlation across the row
A = Ratio2(:,startFrame:endFrame)';
A(isnan(A))=0;
neuronCorrelation = corr(A);
atemp = nancov(A)./sqrt(nanvar(A)'*nanvar(A));
neuronCorrelation(isnan(neuronCorrelation))=atemp(isnan(neuronCorrelation));
neuronCorrelation(isnan(neuronCorrelation))=0;
% 
% for i = 1:neuronNumb
%     for j = 1:neuronNumb
%         %compare i neuron to j neuron
%         corMatrix = nancov(R2_data(i, :), R2_data(j,:))./sqrt(nanvar(R2_data(i,:))'*nanvar(R2_data(j,:)));
%         neuronCorrelation(i,j) = corMatrix(2,1);
%     end
% end


cg = clustergram(neuronCorrelation);
cgIdx = str2double(get(cg,'RowLabels'));
cgIdxRev = cgIdx;
%[~,cgIdxRev]=sort(cgIdx);
%close annoying clustergram plot
%close all hidden

% imagesc(neuronCorrelation)
% xlabel('Neuron')
% ylabel('Neuron')
% title('All Neuron Correlation')
% 
% %Change/display the neuron correlation by cgIdxRev?
% neuronCor_new = neuronCorrelation(cgIdxRev, cgIdxRev); %do I only sort x by cgIdx, or both?
% 
% figure
% imagesc(neuronCor_new)
% xlabel('Sorted Neuron')
% ylabel('Sorted Neuron')
% title('Clustered Neuron Correlation')


end