%% For MFIX
Xin = csvread('/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/tracking_training_data/mfix_training_data/training_feature_props_in.csv');
Xout = csvread('/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/tracking_training_data/mfix_training_data/training_feature_props_out.csv');

% Xin = csvread('/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/tracking_training_data/build/training_feature_props_in.csv');
% Xout = csvread('/Users/sdutta/Codes/fuzzy_rule_based_tracking/fuzzy_tracking_codes/tracking_training_data/build/training_feature_props_out.csv');

%% Train the FIS with training data
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 3;
opt.Verbose = 1;
opt.Exponent = 3
fismat = genfis(Xin,Xout,opt);
showrule(fismat);

%% Plot the membership functions
[x,mf] = plotmf(fismat,'input',1);
subplot(4,1,1), plot(x,mf,'LineWidth',2);
xlabel('Membership Functions for Velocity');

[x,mf] = plotmf(fismat,'input',2);
subplot(4,1,2),plot(x,mf,'LineWidth',2); 
xlabel('Membership Functions for Mass');

[x,mf] = plotmf(fismat,'input',3);
subplot(4,1,3), plot(x,mf,'LineWidth',2);
xlabel('Membership Functions for Volume');

[x,mf] = plotmf(fismat,'input',4);
subplot(4,1,4), plot(x,mf,'LineWidth',2);
xlabel('Membership Functions for Centroid');

% %% Evaluate new input data with the trained system
% xx = [0.326475,0.609921,0.538513,0.32315];
% [output, IRR, ORR, ARR] = evalfis(xx,fismat);

%% Write output data
inputmfs = getfis(fismat,'inmfparams');
inputmfs = inputmfs(:,1:2);

outputmfs = getfis(fismat,'outmfparams');
outputmfs = outputmfs(1,:); % to get one row , since all rows are same.. is it always like this?

dlmwrite('/Users/sdutta/Codes/fuzzy_rule_based_system/fuzzy_rule_based_training/inputParams.txt',inputmfs);
dlmwrite('/Users/sdutta/Codes/fuzzy_rule_based_system/fuzzy_rule_based_training/outputParams.txt',outputmfs);