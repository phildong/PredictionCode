function [corList,R_SquareList] = runCorrelation_AVANeuron(startFrame, endFrame, correlationNeuron)
%split data if it is moving/immobile- correalte only moving to moving, etc.
%correlationNeuron is the AVA neuron we want to compare to

%For data set 110803 (moving only)- AVA is 33 and 16, frames 1-1465

%For data set 102246 (moving-immobile)- AVA is 18 and 30- Moving 1-1590,
%Immobile 1700-4016 (maybe less/still immobile then)

%For data set 111126 (moving-immobile)- AVA is 28 and maybe 64- moving
%1-1785, Immobile(1900-3360)

%R = corrcoef(A,B) returns coefficients between two random variables A and B.


load('heatData.mat');
%Use Ratio2

R2_data = Ratio2(:,startFrame:endFrame);
sz = size(R2);
corList = zeros(sz(1),1);
R_SquareList = zeros(sz(1),1);
R2_AVA = R2_data(correlationNeuron, :);


for i = 1:sz(1)
    
    corMatrix = nancov(R2_AVA, R2_data(i, :))./sqrt(nanvar(R2_AVA)'*nanvar(R2_data(i,:)));
    corList(i) = corMatrix(2,1);
    R_SquareList(i)= corList(i)^2;

end

%Make a bar graph of the correlation

bar(corList)
xlabel('Neuron Number')
labelName = ['Correlation with AVA' num2str(correlationNeuron)];
ylabel(labelName)
grid on 
xticks(0:5:sz(1))

end