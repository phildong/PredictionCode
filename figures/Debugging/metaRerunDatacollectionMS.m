
folder ={}
folder{1}='/projects/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620_XP/';
folder{2}='/projects/LEIFER/PanNeuronal/2017/20170610/BrainScanner20170610_105634_XP/';
folder{3}='/projects/LEIFER/PanNeuronal/2017/20170613/BrainScanner20170613_134800_XP/';
folder{4}='/projects/LEIFER/PanNeuronal/2018/20180709/BrainScanner20180709_100433_XP/';
folder{5}='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_105254_XP/';
folder{6}='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_XP/';
folder{7}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_151024_XP/';
folder{8}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_153839_XP/';
folder{9}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_162140_XP/';
folder{10}='/projects/LEIFER/PanNeuronal/20200310/BrainScanner20200310_141211_XP/';
folder{11}='/projects/LEIFER/PanNeuronal/20200310/BrainScanner20200310_142022_XP/';

failed={};
err={};
j=0;
for k=1:length(folder)
    try
        rerunDataCollectionMS(folder{k})
        load([folder{k}, 'positionDataMS.mat'])
        figure;  
        plot(xPos, yPos); 
        axis equal; 
        title(folder{k})
    catch exception
        j=j+1;
        failed{j}=folder{k}
        err{j}=getReport(exception)
        continue
    end
end
