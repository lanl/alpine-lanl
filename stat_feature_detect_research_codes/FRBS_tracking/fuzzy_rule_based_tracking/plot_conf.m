% plot( x, y, '+r' ); % plot the original points
% n = numel(x); % number of original points
% xi = interp1( 1:n, x, linspace(1, n, 10*n) ); % new sample points 
% yi = interp1(   x, y, xi );
% hold all;
% plot( xi, yi ); % should be smooth between the original points

x = textread('isabel_conf_val_file.txt');
y = (1:1:47);
plot(x);
