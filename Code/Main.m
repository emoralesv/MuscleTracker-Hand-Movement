%%
% PhD. E. Morales-Vargas
% Code to perform the experiments for the classification of sEMG signals.

%% Reading the data
clear all
ds{1} = tabularTextDatastore("C:\Users\L03539078\Documents\GitHub\MuscleTracker\Data\Raw\Biopac\","FileExtensions",".txt"); % reading BIOPACK data
ds{2} = tabularTextDatastore("C:\Users\L03539078\Documents\GitHub\MuscleTracker\Data\Raw\Miotracker\","FileExtensions",".txt"); % reading MIOTRACKER data


%% Initializing the variables
clc
disp("Initializing the variables");

trainWith = [1 2]; % databases to experiment, [1] train with database 1, [2] train with database 2, [1 2] train with the two databases for cross database experiments
numfeatures = 60;

n =  zeros(1,size(ds,2));
for i = 1 : size(ds,2), n(i) = size(ds{i}.Files,1); end % number of samples for databases
spatialRes = []; normType ={}; channel = {}; model = {}; accDatabase1 = []; accDatabase2 = [];  accBoth = []; % variables to store the results
results = table(spatialRes,normType,channel,model,accDatabase1,accDatabase2,accBoth); % table to store the results
mn = min(n); % balancing the databases
% index to calculate results by database

perm =  zeros(mn,size(ds,2));
for p = 1: size(ds,2), perm(:,p) = randperm(n(p),mn); end

data.Files = {};
data.idxn = [];
data.idx = [];
for d = trainWith, data.Files = [data.Files; ds{d}.Files(perm(:,d))]; data.idxn = [data.idxn; repmat(d,[n(d) 1])]; end % generate final database


fprintf("Loaded %i samples \n",size(data.Files,1));
disp("Done without errors");
%% filter design
%Butter filter
fs = 1001;
[A,B,i,D] = butter(4,[20 500]/(fs/2));
filt1 = designfilt("bandpassiir",FilterOrder=4, ...
    HalfPowerFrequency1=20, ...
    HalfPowerFrequency2=500, ...
    SampleRate=fs);
%fvt = fvtool(filt1,Fs=fs);
%legend(fvt,"Filter");

% Notch filter
notchSpecs = fdesign.notch('N,F0,Q',4,60,30,1000);
filt = design(notchSpecs,'Systemobject',true);
%fvtool(filt)

%% Experimenter loop
c = 1;
for  spatialSamples = 3

    feat1 = zeros(size(data.Files,1),(numfeatures)+1);
    feat2 = zeros(size(data.Files,1),(numfeatures)+1);
    featcombined = zeros(size(data.Files,1),1+((numfeatures)*2));

    feat1_f = zeros(size(data.Files,1),(numfeatures)+1);
    feat2_f = zeros(size(data.Files,1),(numfeatures)+1);
    featcombined_f = zeros(size(data.Files,1),1+((numfeatures)*2));

    feat1_n = zeros(size(data.Files,1),(numfeatures)+1);
    feat2_n = zeros(size(data.Files,1),(numfeatures)+1);
    featcombined_n = zeros(size(data.Files,1),1+((numfeatures)*2));

    feat1_nf = zeros(size(data.Files,1),(numfeatures)+1);
    feat2_nf = zeros(size(data.Files,1),(numfeatures)+1);
    featcombined_nf = zeros(size(data.Files,1),1+((numfeatures)*2));



    for i = 1: size(data.Files,1)
        fprintf("Processing file %i of %i \n",i,size(data.Files,1));
        name = data.Files{i};
        d = readmatrix(name);
        cc1 = double(d(:,1));
        cc2 = double(d(:,2));
        delta  =  floor (size(cc1,1) / spatialSamples)-1;
        cc = 1;
        for samples = 1: delta:  spatialSamples*delta
            %[url,name,ext] = fileparts(name);  % to save the data
            %writematrix([cc1 cc2],strcat("C:\Users\L03539078\Documents\GitHub\MuscleTracker\Data\Sampled\Miotracker\",name,"_",num2str(cc),ext));
            c1 = cc1(samples:samples+delta);
            c2 = cc2(samples:samples+delta);
            data.idx(c,1) = data.idxn(i);
            % raw features
            feat1(c,:)  =[int_feature(c1) str2double(name(end-4))];
            feat2(c,:)  = [int_feature(c2) str2double(name(end-4))];
            featcombined(c,:)  = [feat1(c,1:end-1) feat2(c,:)];


            % features with the signal filtered
            feat1_f(c,:)  =[int_feature(filt(filter(filt1,c1))) str2double(name(end-4))];
            feat2_f(c,:)  = [int_feature(filt(filter(filt1,c2))) str2double(name(end-4))];
            featcombined_f(c,:)  = [feat1(c,1:end-1) feat2(c,:)];


            % features with the normalized signal zcore and range
            feat1_n(c,:)  =[int_feature(normalize( normalize(c1,"zscore"),"range")) str2double(name(end-4))];
            feat2_n(c,:)  = [int_feature(normalize( normalize(c2,"zscore"),"range")) str2double(name(end-4))];
            featcombined_n(c,:)  = [feat1(c,1:end-1) feat2(c,:)];


            % normalized + filtered
            feat1_nf(c,:)  =[int_feature(normalize(normalize(filt(filter(filt1,c1)),"zscore"),"range")) str2double(name(end-4))];
            feat2_nf(c,:)  = [int_feature(normalize(normalize(filt(filter(filt1,c2)),"zscore"),"range")) str2double(name(end-4))];
            featcombined_nf(c,:)  = [feat1(c,1:end-1) feat2(c,:)];


            c = c +1;
            cc = cc+ +1;
        end
    end


    %% Model training
    results = [results; trKNN(feat1,data.idx,{spatialSamples,"no","ch1", "trKNN"})];
    results = [results; trKNN(feat2,data.idx,{spatialSamples,"no","ch2", "trKNN"})];
    results = [results; trKNN(feat1_n,data.idx,{spatialSamples,"zerocenter","ch1", "trKNN"})];
    results = [results; trKNN(feat2_n,data.idx,{spatialSamples,"zerocenter","ch2", "trKNN"})];
    results = [results; trKNN(feat1_f,data.idx,{spatialSamples,"filtered","ch1", "trKNN"})];
    results = [results; trKNN(feat2_f,data.idx,{spatialSamples,"filtered","ch2", "trKNN"})];
    results = [results; trKNN(feat1_nf,data.idx,{spatialSamples,"normalized+filtered","ch1", "trKNN"})];
    results = [results; trKNN(feat2_nf,data.idx,{spatialSamples,"normalized+filtered","ch2", "trKNN"})];
    results = [results; trKNN(featcombined,data.idx,{spatialSamples,"no","combined", "trKNN"})];
    results = [results; trKNN(featcombined_n,data.idx,{spatialSamples,"zerocenter","combined", "trKNN"})];
    results = [results; trKNN(featcombined_f,data.idx,{spatialSamples,"filtered","combined", "trKNN"})];
    results = [results; trKNN(featcombined_nf,data.idx,{spatialSamples,"normalized+filtered","combined", "trKNN"})];

    results = [results; trSVM(feat1,data.idx,{spatialSamples,"no","ch1", "trSVM"})];
    results = [results; trSVM(feat2,data.idx,{spatialSamples,"no","ch2", "trSVM"})];
    results = [results; trSVM(feat1_n,data.idx,{spatialSamples,"zerocenter","ch1", "trSVM"})];
    results = [results; trSVM(feat2_n,data.idx,{spatialSamples,"zerocenter","ch2", "trSVM"})];
    results = [results; trSVM(feat1_f,data.idx,{spatialSamples,"filtered","ch1", "trSVM"})];
    results = [results; trSVM(feat2_f,data.idx,{spatialSamples,"filtered","ch2", "trSVM"})];
    results = [results; trSVM(feat1_nf,data.idx,{spatialSamples,"normalized+filtered","ch1", "trSVM"})];
    results = [results; trSVM(feat2_nf,data.idx,{spatialSamples,"normalized+filtered","ch2", "trSVM"})];
    results = [results; trSVM(featcombined,data.idx,{spatialSamples,"no","combined", "trSVM"})];
    results = [results; trSVM(featcombined_n,data.idx,{spatialSamples,"zerocenter","combined", "trSVM"})];
    results = [results; trSVM(featcombined_f,data.idx,{spatialSamples,"filtered","combined", "trSVM"})];
    results = [results; trSVM(featcombined_nf,data.idx,{spatialSamples,"normalized+filtered","combined", "trSVM"})];

    results = [results; trNN(feat1,data.idx,{spatialSamples,"no","ch1", "trNN"})];
    results = [results; trNN(feat2,data.idx,{spatialSamples,"no","ch2", "trNN"})];
    results = [results; trNN(feat1_n,data.idx,{spatialSamples,"zerocenter","ch1", "trNN"})];
    results = [results; trNN(feat2_n,data.idx,{spatialSamples,"zerocenter","ch2", "trNN"})];
    results = [results; trNN(feat1_f,data.idx,{spatialSamples,"filtered","ch1", "trNN"})];
    results = [results; trNN(feat2_f,data.idx,{spatialSamples,"filtered","ch2", "trNN"})];
    results = [results; trNN(feat1_nf,data.idx,{spatialSamples,"normalized+filtered","ch1", "trNN"})];
    results = [results; trNN(feat2_nf,data.idx,{spatialSamples,"normalized+filtered","ch2", "trNN"})];
    results = [results; trNN(featcombined,data.idx,{spatialSamples,"no","combined", "trNN"})];
    results = [results; trNN(featcombined_n,data.idx,{spatialSamples,"zerocenter","combined", "trNN"})];
    results = [results; trNN(featcombined_f,data.idx,{spatialSamples,"filtered","combined", "trNN"})];
    results = [results; trNN(featcombined_nf,data.idx,{spatialSamples,"normalized+filtered","combined", "trNN"})];

    results = [results; trSVMCubic(feat1,data.idx,{spatialSamples,"no","ch1", "trSVMCubic"})];
    results = [results; trSVMCubic(feat2,data.idx,{spatialSamples,"no","ch2", "trSVMCubic"})];
    results = [results; trSVMCubic(feat1_n,data.idx,{spatialSamples,"zerocenter","ch1", "trSVMCubic"})];
    results = [results; trSVMCubic(feat2_n,data.idx,{spatialSamples,"zerocenter","ch2", "trSVMCubic"})];
    results = [results; trSVMCubic(feat1_f,data.idx,{spatialSamples,"filtered","ch1", "trSVMCubic"})];
    results = [results; trSVMCubic(feat2_f,data.idx,{spatialSamples,"filtered","ch2", "trSVMCubic"})];
    results = [results; trSVMCubic(feat1_nf,data.idx,{spatialSamples,"normalized+filtered","ch1", "trSVMCubic"})];
    results = [results; trSVMCubic(feat2_nf,data.idx,{spatialSamples,"normalized+filtered","ch2", "trSVMCubic"})];
    results = [results; trSVMCubic(featcombined,data.idx,{spatialSamples,"no","combined", "trSVMCubic"})];
    results = [results; trSVMCubic(featcombined_n,data.idx,{spatialSamples,"zerocenter","combined", "trSVMCubic"})];
    results = [results; trSVMCubic(featcombined_f,data.idx,{spatialSamples,"filtered","combined", "trSVMCubic"})];
    results = [results; trSVMCubic(featcombined_nf,data.idx,{spatialSamples,"normalized+filtered","combined", "trSVMCubic"})];

    results = [results; trSVMQuadratic(feat1,data.idx,{spatialSamples,"no","ch1", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat2,data.idx,{spatialSamples,"no","ch2", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat1_n,data.idx,{spatialSamples,"zerocenter","ch1", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat2_n,data.idx,{spatialSamples,"zerocenter","ch2", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat1_f,data.idx,{spatialSamples,"filtered","ch1", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat2_f,data.idx,{spatialSamples,"filtered","ch2", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat1_nf,data.idx,{spatialSamples,"normalized+filtered","ch1", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(feat2_nf,data.idx,{spatialSamples,"normalized+filtered","ch2", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(featcombined,data.idx,{spatialSamples,"no","combined", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(featcombined_n,data.idx,{spatialSamples,"zerocenter","combined", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(featcombined_f,data.idx,{spatialSamples,"filtered","combined", "SVMCuadratic"})];
    results = [results; trSVMQuadratic(featcombined_nf,data.idx,{spatialSamples,"normalized+filtered","combined", "SVMCuadratic"})];








writetable(results,"results");
end





function [f] = int_feature(x)
f = [];
  
feat = {
    'msr'    ,...
    'lcov'  ,...
    'fzc',...
    'ewl'    ,...
    'emav'   ,...
    'asm'    ,...
    'ass'  ,...
    'ltkeo'  ,...
    'card'   ,...
    'ldasdv' ,...
    'ldamv'  ,...
    'dvarv'  , ...
    'mfl'   ,...
    'myop'   ,...
    'ssi'    ,...
    'vo'     ,... 
    'tm'    ,...
    'aac'   ,...
    'mmav'   ,...
    'mmav2'  ,...
    'iemg'  ,...
    'dasdv'  ,...
    'damv'   ,...
    'rms'    , ...
    'vare'   ,...
    'wa'     ,...
    'ld'     ,...
    'ar'     ,...
    'mav'    ,...
    'zc'     ,...
    'ssc'    ,...
    'wl'    ,...
    'mad'    ,...
    'iqr'    , ...
    'kurt'   ,...
    'skew'  ,...
    'cov'    ,...
    'sd'     ,...
    'var'    ,...
    'ae'    };

L = 100;
for ff = feat
    fn= jfemg(ff{1},x);
    f = [f real(fn)];
end

    Y = fft(x);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

   decomplevels = 6;
    wv = "db2";
    [c,l] = wavedec(x,decomplevels,wv);
    [Ea,Ed] = wenergy(c,l);

    t = wpdec(x,3,wv);
    e = wenergy(t);

    f = [mean(P1), median(P1) max(P1) Ed e f];  

end
















