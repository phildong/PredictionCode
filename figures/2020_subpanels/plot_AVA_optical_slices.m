file='AVAR_RFP.mat'
dir='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_XP/user_exported_tracking_frames/'
blue_cmap = colorcet('L15');
red_cmap = colorcet('L13');
curr_cmap= red_cmap;

load([dir, file])
figure('Position', [100 100 1000 700])
imagesc(baseImg)
hold on
x=closeXY(:,1);
y=closeXY(:,2);
plot(x, y,'ow')
plot(inSliceXY(:,1),inSliceXY(:,2),'xw')
set(gca,'YDir','normal')
dx=0
dy=3
mat2clust_pyindx=[56,33,103,8,28,118,115,38,96,9,126,88,70,110,53,130,15,62,66,113,84,35,121,86,111,11,82,65,75,67,36,40,132,42,25,55,94,106,23,48,122,87,18,2,71,124,27,83,31,78,93,34,123,41,51,61,5,120,73,58,64,99,74,26,7,50,81,44,91,72,10,14,68,20,89,133,24,43,95,3,131,69,49,57,104,21,77,30,117,1,105,22,92,54,6,98,47,97,39,129,60,90,17,16,100,101,37,119,32,127,63,79,85,29,4,102,125,0,109,76,46,80,114,12,19,112,45,128,107,116,13,52,108,59];
cell_numeric_labels=num2cell(mat2clust_pyindx(closePointsIds));
labels=cellfun(@num2str,cell_numeric_labels,'UniformOutput',false)
text(x+dx, y+dy, labels, 'color','white');
xlim([25, 400])
ylim([150, 400])
title(file)

cmap=colormap(curr_cmap);


