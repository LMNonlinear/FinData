function [ b_sig, b_nu ] = real_scaling_tLoc(daily_returns, dist_type)
%Fits various models to data of scaled distributions to determine best one

emp_dist1 = fitdist(daily_returns,dist_type);

runs = 49;
act_sigmas = zeros(1,runs);
act_nus = zeros(1,runs);
scales = zeros(1,runs);

for j = 2:runs+1
    
    scale = j;
    returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),scale)+1:...
        end),scale,[]))';
    emp_dist2 = fitdist(returns_2,dist_type);
    
    act_sigmas(j-1) = emp_dist2.sigma;
    act_nus(j-1) = emp_dist2.nu;
    
    scales(j-1) = j;
end
act_nus(act_nus == max(act_nus)) = max(act_nus)/2;
act_nus(act_nus == max(act_nus)) = max(act_nus)/2;

% sigmas - alpha fit
lscales = log(scales);
lsigmas = log(act_sigmas);
b_sig = lscales'\(lsigmas+4.8)';
figure()
hold on
plot(lscales,lsigmas)
plot(lscales,-4.8+b_sig*lscales)
plot(lscales,log(emp_dist1.sigma)+0.5*lscales)
ylim([-4.4,-2.4])
title('Sigma as a function of scale size of grouped returns')
xlabel('log(scaling size (in days))')
ylabel('log(sigma)')
legend({'empirical sigmas','log-linear fit of sigmas',...
    'theoretical sigmas'},'location','NorthWest')
b_sig = [-4.8,b_sig];

% nus
% linear fit (best poss even if is shit)
lags = [ones(1,length(scales)); scales];
b_nu = lags'\(act_nus)';
figure()
hold on
plot(scales,act_nus)
plot(scales,b_nu(1)+b_nu(2)*scales)
ylim([3,9])
title('Nu as a function of scale size of grouped returns')
xlabel('scaling size (in days)')
ylabel('nu')
legend({'empirical nus','linear fit of nus'},'location','NorthWest')


end