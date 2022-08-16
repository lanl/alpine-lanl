%% Read training data
Xin = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_in.csv');
Xout = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_out.csv');
Xintest = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_in.csv'); %just for testing with matlab inference.
opt = NaN(4,1);
opt(4) = 0;
cluster_num = 3; % User given parameter decides no of rules

%% Train the FIS with training data
fismat = genfis3_new(Xin,Xout,'sugeno',cluster_num,opt);


%% Plot the membership functions
[x,mf] = plotmf(fismat,'input',1);
subplot(4,1,1), plot(x,mf);
xlabel('Membership Functions for Velocity');

[x,mf] = plotmf(fismat,'input',2);
subplot(4,1,2),plot(x,mf); 
xlabel('Membership Functions for Mass');

[x,mf] = plotmf(fismat,'input',3);
subplot(4,1,3), plot(x,mf);
xlabel('Membership Functions for Volume');

[x,mf] = plotmf(fismat,'input',4);
subplot(4,1,4), plot(x,mf);
xlabel('Membership Functions for Centroid');

%% Evaluate new input data with the trained system
xx = [0.326475,0.609921,0.538513,0.32315];
[output, IRR, ORR, ARR] = evalfis(xx,fismat);

%% Write output data
inputmfs = getfis(fismat,'inmfparams');
inputmfs = inputmfs(:,1:2);

outputmfs = getfis(fismat,'outmfparams');
outputmfs = outputmfs(1,:); % to get one row , since all rows are same.. is it always like this?

dlmwrite('/home/soumya/Dropbox/Codes/fuzzy_rule_based_tracking/inputmfs.txt',inputmfs);
dlmwrite('/home/soumya/Dropbox/Codes/fuzzy_rule_based_tracking/outputmfs.txt',outputmfs);