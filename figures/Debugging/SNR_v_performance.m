%Adding path for dependencies
addpath(genpath('/projects/LEIFER/communalCode/3dbrain'))

recording{1}='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_110803_test_new/sCMOS_Frames_U16_1024x512.dat';
ID{1}='BrainScanner20200130_110803_test_new';

recording{2}='/projects/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620_xinwei/sCMOS_Frames_U16_1024x512.dat';
ID{2}='BrainScanner20170424_105620_xinwei';

recording{3}='/projects/LEIFER/PanNeuronal/20200130/BrainScanner20200130_105254_test_new/sCMOS_Frames_U16_1024x512.dat';
ID{3}='BrainScanner20200130_105254_test_new';

recording{4}='/projects/LEIFER/PanNeuronal/20200310/BrainScanner20200310_142022/sCMOS_Frames_U16_1024x512.dat';
ID{4}='BrainScanner20200310_142022';

recording{5}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_151024/sCMOS_Frames_U16_1024x512.dat';
ID{5}='BrainScanner20200309_151024';

recording{6}='/projects/LEIFER/PanNeuronal/20200310/BrainScanner20200310_141211/sCMOS_Frames_U16_1024x512.dat';
ID{6}='BrainScanner20200310_141211';

recording{7}='/projects/LEIFER/PanNeuronal/2017/20170613/BrainScanner20170613_134800/sCMOS_Frames_U16_1024x512.dat';
ID{7}='BrainScanner20170613_134800';

recording{8}='/projects/LEIFER/PanNeuronal/2018/20180709/BrainScanner20180709_100433/sCMOS_Frames_U16_1024x512.dat';
ID{8}='BrainScanner20180709_100433';

recording{9}='/projects/LEIFER/PanNeuronal/2017/20170610/BrainScanner20170610_105634/sCMOS_Frames_U16_1024x512.dat';
ID{9}='BrainScanner20170610_105634';

recording{10}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_153839/sCMOS_Frames_U16_1024x512.dat';
ID{10}='BrainScanner20200309_153839';

recording{11}='/projects/LEIFER/PanNeuronal/20200309/BrainScanner20200309_162140/sCMOS_Frames_U16_1024x512.dat';
ID{11}='BrainScanner20200309_162140';

%Loop through files and
%Check that all the files exist
for k=1:length(recording)
    assert(isfile(recording{k}))
end
disp('Files exist');

%Loop through files again and collect pixel level statistics

for k=1:length(recording)
    disp('Starting to analyze a new recording...');
    disp(recording{k});
    [imMean{k}, imStd{k}, cumStd{k}] = datPixelStatistics(recording{k},1024,512);
    cumMean{k}=mean(imMean{k});
end

figure;
plot([1:11], cell2mat(cumStd),'o')
ylabel('cumStd'); xlabel('rho^2 rank')

figure;
plot([1:11], cell2mat(cumStd)./cell2mat(cumMean),'o')
ylabel('cumStd/cumMean'); xlabel('rho^2 rank')

figure;
plot([1:11], cell2mat(cumMean),'o')
ylabel('cumMean'); xlabel('rho^2 rank')