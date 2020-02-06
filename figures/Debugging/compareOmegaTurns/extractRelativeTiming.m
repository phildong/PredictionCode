% These are the relevant timepoints on the VLC video of the brightfield
SpM = 60;
START = 1;
FINISH = 2;
vlc_bf(START) = 13 * SpM + 21; %in vlc seconds
vlc_bf(FINISH) = 13 * SpM + 57; %in vlc seconds

vlc_bf_FrameRate = 30.0003; %frames / vlc second
acquired_bf_Framerate = 50; %frames /aquired second


bf = vlc_bf*30.0003; % brightfield frame number


%Get the time alignment between brightfield and hi magnification images
% by comparing flashes
[bfAll, ~, hiResData] =  tripleFlashAlign('/tigress/LEIFER/PanNeuronal/2017/20170610/BrainScanner20170610_105634');

%
commonTime(START) = bfAll.frameTime(round(bf(START)))
commonTime(FINISH) = bfAll.frameTime(round(bf(FINISH)))

volume = round(interp1(hiResData.frameTime,hiResData.stackIdx,commonTime,'linear'))

