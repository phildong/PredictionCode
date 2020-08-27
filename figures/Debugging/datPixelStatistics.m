function  [imMean, imStd, cumStd] =datPixelStatistics(datFile,rows,cols,rowSearch)
% function takes a .dat file for worm data and searches each of the images
% for a flash. The images must be rowxcol matrix of uint16, works super
% fast. Can also select input rows. This will find the flash signal by
% averaging over the first rows in order to speed up program. The flash is
% normally bright enough that I only need 1 row or less...

%% load images and find flash in images using user defined ROI
if nargin==0
    datFile=uipickfiles;
    datFile=datFile{1};
end

if nargin<3
    rows=1200;
    cols=600;
end
if nargin<4
    rowSearch=cols;
end



if strfind(datFile, '.dat');   
    Fid=fopen(datFile);
status=fseek(Fid,0,1);
totNumImages=floor(ftell(Fid)/(2*rows*cols)-1);
status=fseek(Fid,0,-1);
else
    error('File must be a .dat binary file in uint16 form');
end


if 1
pixelValues=fread(Fid,rows*cols,'uint16',0,'l');
initialIm=reshape(pixelValues,rows,cols);



fig=imagesc(initialIm);
display('Select area to search');
roiFlash=roipoly;
delete(fig)

end


%% search for the flash
imMean=zeros(1,totNumImages);
%cell1=imMean;
%cell2=imMean;
progressbar(0)
for iFrame=1:totNumImages
    progressbar(iFrame/totNumImages)
    pixelValues=fread(Fid,rows*cols,'uint16',0,'l');
    status=fseek(Fid,2*cols*(rows)*iFrame,-1);
    N=0; S1=0; S2=0;
    
    X=pixelValues(roiFlash);
    
    %Calculate mean and std for each image
    imMean(iFrame)=mean(X);
    imStd(iFrame)=std(X);

    %Calculate a running tally of standard deviation
    N=N+numel(X);
    S1=S1+sum(X);
    S2=S2+sum(X.^2);
end
cumStd=sqrt( N*S2- S1.^2 )/N;

%%
fclose(Fid);
