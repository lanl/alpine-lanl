%% Read training data
load iris.dat
Xin = iris(:,1:4);
Xout = iris(:,5);

% Xin = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_in.csv');
% Xout = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_out.csv');
% Xintest = csvread('/home/soumya/Dropbox/Codes/tracking_training_data/build/training_feature_props_in.csv'); %just for testing with matlab inference.
opt = NaN(4,1);
opt(4) = 0;
cluster_num = 3; % User given parameter

%% Train the FIS with training data
fismat = genfis3_new(Xin,Xout,'sugeno',cluster_num,opt);

%% Plot the membership functions
[x,mf] = plotmf(fismat,'input',1);
subplot(4,1,1), plot(x,mf);
xlabel('Membership Functions for Input 1');

[x,mf] = plotmf(fismat,'input',2);
subplot(4,1,2),plot(x,mf); 
xlabel('Membership Functions for Input 2');

[x,mf] = plotmf(fismat,'input',3);
subplot(4,1,3), plot(x,mf);
xlabel('Membership Functions for Input 3');

[x,mf] = plotmf(fismat,'input',4);
subplot(4,1,4), plot(x,mf);
xlabel('Membership Functions for Input 4');


%% Evaluate new input data with the trained system
% ARR = firing strengths
% IRR = input evalueted membership funcs
%test_ip = [0.00599523,0.086754,0.0922562,0.00552427];
[output, IRR, ORR, ARR] = evalfis(Xin(101:102,1:4),fismat);
disp(output)

% %% Write output data
% inputmfs = getfis(fismat,'inmfparams');
% inputmfs = inputmfs(:,1:2);% 
outputmfs = getfis(fismat,'outmfparams');
disp(outputmfs)
% outputmfs = outputmfs(1,:); % to get one row , since all rows are same.. is it always like this?% 
% dlmwrite('/home/soumya/Dropbox/Codes/fuzzy_rule_based_tracking/inputmfs.txt',inputmfs);
% dlmwrite('/home/soumya/Dropbox/Codes/fuzzy_rule_based_tracking/outputmfs.txt',outputmfs);