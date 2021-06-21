file='AVAL_RFP.mat'
dir='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_XP/user_exported_tracking_frames/'
blue_cmap = colorcet('L15');
red_cmap = colorcet('L13');
if contains(file, 'BFP')
    curr_cmap= blue_cmap;
else
    curr_cmap=red_cmap;
end

load([dir, file])
figure('Position', [50 50 550 200])
imagesc(baseImg)
hold on
x=closeXY(:,1);
y=closeXY(:,2);
plot(x, y,'ow')
plot(inSliceXY(:,1),inSliceXY(:,2),'+w')
set(gca,'YDir','normal')
dx=0
dy=3
%Note each element is the clustered python  0-indexed neuron number
%used displayed in the manuscript text and heatmaps
% so for the 4th neuron in matlab (1-index), the 4th element of the
% array gives the neuron number used in the manuscript
mat2clust_pyindx=[97,36,70,66,10,14,26,12,51,67,108,38,33,100,99,107,60,62,16,88,56,41,63,28,86,15,59,123,5,29,42,64,110,9,6,11,121,55,46,68,126,23,113,132,133,104,19,103,21,79,118,39,127,130,128,96,3,44,92,78,47,17,93,8,131,7,124,76,72,85,18,24,20,40,37,111,73,115,119,0,109,1,94,112,69,98,80,74,125,4,101,50,65,27,90,54,30,122,13,105,34,35,48,61,81,45,43,82,83,106,75,31,25,114,2,71,91,49,95,53,117,32,57,87,22,89,77,102,120,52,58,129,84,116]
cell_numeric_labels=num2cell(mat2clust_pyindx(closePointsIds));
labels=cellfun(@num2str,cell_numeric_labels,'UniformOutput',false)
text(x+dx, y+dy, labels, 'color','white');
axis image
%xlim([60, 350])
%ylim([220, 330])
d=47.3;
plot([200, 200+d],[250, 250],'w') % using calibration 473px/mm this bar should be 100 microns
plot([150, 150],[225, 225+d],'w') % using calibration 473px/mm this bar should be 100 microns
title(file)
set(gca,'xtick',[])
set(gca,'ytick',[])
cmap=colormap(curr_cmap);
camroll(-28)
camtarget([213, 267, 0])
camzoom(9)

