%% Data is generated using a python notebook

num_samples = 200
in_data = 'gaussian_data_in.csv'
out_data = 'gaussian_data_out.csv'
samples = table2array(readtable(in_data));
out_val = table2array(readtable(out_data));


%% Train the FIS with training data
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 2;
opt.Verbose = 0;
opt.Exponent = 4
fismat = genfis(samples,out_val,opt);
showrule(fismat);

%% Plot the membership functions
[x,mf] = plotmf(fismat,'input',1);
subplot(2,1,1), plot(x,mf,'LineWidth',3);
set(gca,'xtick',[]) % to remove ticks
set(gca,'ytick',[])% to remove ticks
%xlabel('Membership Functions for input1','FontSize' ,25);

[x,mf] = plotmf(fismat,'input',2);
subplot(2,1,2),plot(x,mf,'LineWidth',3); 
set(gca,'xtick',[])% to remove ticks
set(gca,'ytick',[])% to remove ticks
%xlabel('Membership Functions for input2','FontSize' ,15);

%% Evaluate new input data with the trained system
xx = [7.5,2.5];
[output, IRR, ORR, ARR] = evalfis(xx,fismat);

% % show the training points
% colormap(jet);
% a = zeros(1,num_samples/2);
% b = ones(1,num_samples/2);
% c = [a,b];
% subplot(3,1,1)
% scatter(samples(:,1),samples(:,2),25,c,'filled','MarkerFaceAlpha',1.0);
% set(gca,'xtick',[])% to remove ticks
% set(gca,'ytick',[])% to remove ticks
% % title('2D data points','FontSize' ,30);
% % xlabel('Input 1','FontSize' ,25);
% % ylabel('Input 2','FontSize' ,25);