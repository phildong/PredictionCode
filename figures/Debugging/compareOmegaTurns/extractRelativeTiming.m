%Load the spreadsheet
csvfile = '/home/leifer/workspace/PredictionCode/figures/Debugging/compareOmegaTurns/EscapeResponseTimePoints.csv';
T = readtable(csvfile,'Format','%s%s%f%u%u%u%u%s','Delimiter',',')



vlc_bf_FrameRate = T.FPS; %frames / vlc second

% These are the relevant timepoints on the VLC video of the brightfield
vlc_bf = [T.StartMin*60 + T.StartSec, T.EndMin * 60 + T.EndSec ]; %in vlc seconds

for k=1:size(vlc_bf,1)
    bf(k,:) = vlc_bf(k,:).*vlc_bf_FrameRate(k); % brightfield frame number
    %Get the time alignment between brightfield and hi magnification images
    
    % by comparing flashes
    [bfAll, ~, hiResData] =  tripleFlashAlign(T.BrainScannerFolder{k});

    commonTime(k,:) = [bfAll.frameTime(round(bf(k, START))), bfAll.frameTime(round(bf(k,FINISH)))];
    
    %skip repeated time stamps for interpolation otherwise interpolation
    %will error
    ind = find(diff(hiResData.frameTime)~=0);
    volume(k,:) = round(interp1(hiResData.frameTime(ind),hiResData.stackIdx(ind),commonTime(k,:),'linear'));

end

U=[T,array2table(volume,'VariableNames',{'StartVolume','EndVolume'})];
writetable(U,'realtiveTimingWithVolumes.txt','Delimiter',',') 


