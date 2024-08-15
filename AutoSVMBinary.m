clc;
clear all;
% normal IROI features
OverallFeaturesTrain=xlsread('C:\Users\ayush\OneDrive\Desktop\Project data\Combinations tried\Laws 3 iroi + laws 5 ratio train.xlsx','A1:DA214');
OverallFeaturesTest=xlsread('C:\Users\ayush\OneDrive\Desktop\Project data\Combinations tried\Laws 3 iroi + laws 5 ratio test.xlsx','A1:DA60');

% concatenated features 
% OverallFeaturesTrain=xlsread("C:\Users\PLC Lab1\Downloads\JYOTI DIXIT\SROI project 1 May\Gabor Filters\all\gabor_concate_features_train.xlsx",'A1:CF310');
% OverallFeaturesTest=xlsread("C:\Users\PLC Lab1\Downloads\JYOTI DIXIT\SROI project 1 May\Gabor Filters\all\gabor_concate_features_test.xlsx",'A1:CF90');

% ratio features
% OverallFeaturesTrain=xlsread("C:\Users\PLC Lab1\Downloads\JYOTI DIXIT\SROI project 1 May\Gabor Filters\all\gabor_ratio_features_train.xlsx",'A1:AP310');
% OverallFeaturesTest=xlsread("C:\Users\PLC Lab1\Downloads\JYOTI DIXIT\SROI project 1 May\Gabor Filters\all\gabor_ratio_features_test.xlsx",'A1:AP90');

[OverallFeaturesTrainN,ps] = mapminmax(OverallFeaturesTrain',0,1);
OverallFeaturesTrainN = double(OverallFeaturesTrainN');
OverallFeaturesTestN = mapminmax('apply',OverallFeaturesTest',ps);
OverallFeaturesTestN = double(OverallFeaturesTestN');
% TrainLabel=[zeros(1,22),ones(1,31)]';
% TestLabel=[zeros(1,22),ones(1,31)]';

% Train Label ~ vector concatenation
% train label for normal IROI & ratio features , concatenated features
TrainLabel = [zeros(109,1); ones(105,1)];

% TrainLabel=zeroes(1770,1); 
% TrainLabel(1:590)=0;
% TrainLabel(591:1180)=1;
% TrainLabel(1181:1770)=2;

% Test Label ~ normal IROI,Ratio features & concatrenated features
TestLabel = [zeros(30,1); ones(30,1)];

% % TestLabel = zeros(90, 1);  
% TestLabel(1:30) = 0;       
% Testabel(31:60) = 1;     
% TestLabel(60:90) = 2; 

bestcv = 0;
bestc=1;
bestg=1;
% grid search on libsvm
for log2c = -3:14,
for log2g = -12:4,
     cmd = ['-v 10 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
cv = svmtrain(TrainLabel,OverallFeaturesTrainN,cmd);
if (cv >= bestcv),
bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
end
fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv)
end
end
% bestc;
% bestg; 
cval=10;
Parameters=sprintf('-c %g -g %g -v %g',bestc,bestg,cval);
ParamFinal=sprintf('-c %g -g %g',bestc,bestg);
Model=svmtrain(TrainLabel,OverallFeaturesTrainN,Parameters);
FinalModel=svmtrain(TrainLabel,OverallFeaturesTrainN,ParamFinal);
[PredictedLabelOverallFeatures TestAccuracyOverallFeatures ProbabilityOverallFeatures]=svmpredict(TestLabel,OverallFeaturesTestN,FinalModel);

PredictedLabelOverallFeatures = svmpredict(TestLabel, OverallFeaturesTestN, FinalModel);

% Calculate confusion matrix elements ~ normal IROI & ratio features
% M11 = length(find(PredictedLabelOverallFeatures(1:30,:) == 0));
% M12 = length(find(PredictedLabelOverallFeatures(1:30,:) == 1));
% M13 = length(find(PredictedLabelOverallFeatures(1:30,:) == 2));
% M21 = length(find(PredictedLabelOverallFeatures(31:60,:) == 0));
% M22 = length(find(PredictedLabelOverallFeatures(31:60,:) == 1));
% M23 = length(find(PredictedLabelOverallFeatures(31:60,:) == 2));
% M31 = length(find(PredictedLabelOverallFeatures(61:90,:) == 0));
% M32 = length(find(PredictedLabelOverallFeatures(61:90,:) == 1));
% M33 = length(find(PredictedLabelOverallFeatures(61:90,:) == 2));


% Calculate confusion matrix elements ~ Concatenated features

% M11 = length(find(PredictedLabelOverallFeatures(1:60,:) == 0));
% M12 = length(find(PredictedLabelOverallFeatures(1:60,:) == 1));
% M13 = length(find(PredictedLabelOverallFeatures(1:60,:) == 2));
% M21 = length(find(PredictedLabelOverallFeatures(61:120,:) == 0));
% M22 = length(find(PredictedLabelOverallFeatures(61:120,:) == 1));
% M23 = length(find(PredictedLabelOverallFeatures(61:120,:) == 2));
% M31 = length(find(PredictedLabelOvrerallFeatures(121:180,:) == 0));
% M32 = length(find(PredictedLabelOrverallFeatures(121:180,:) == 1));
% M33 = length(find(PredictedLabelOverallFeatures(121:180,:) == 2));

% Confusion matrix for 3 labels
% CM = [M11 M12 M13; M21 M22 M23; M31 M32 M33]

M11=length(find(PredictedLabelOverallFeatures(1:30,:)==0));
M12=length(find(PredictedLabelOverallFeatures(1:30,:)==1));
M21=length(find(PredictedLabelOverallFeatures(31:60,:)==0));
M22=length(find(PredictedLabelOverallFeatures(31:60,:)==1));
CM=[M11 M12;M21 M22]
