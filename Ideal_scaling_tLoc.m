%create pure tLocScale data for testing theory
%maybe use this to show errors on model by bootstrapping to find standard
%variation in params and show that your predicitions lie within this error

x = 0.0001:0.0001:1-0.0001;
pd1 = fitdist(daily_returns,dist_type);

% This is ideal starting data
y = icdf(pd1,x)';

%calc_sigmas = zeros(1,100);
act_sigmas = zeros(1,100);
lags = zeros(1,100);
act_nus = zeros(1,100);

for j = 2:101
    
    lag2 = j;
    sigmas = zeros(1,100);
    nus = zeros(1,100);
    for i = 1:100
        y = y(randperm(length(y)));
        
        % Ideal scaled data
        y2 = sum(reshape(y(rem(length(y),lag2)+1:end),lag2,[]))';
        pd2 = fitdist(y2,dist_type);
        sigmas(i) = pd2.sigma;
        nus(i) = pd2.nu;
    end
    
    %calc_sigmas(j) = mean(means)-pd1.beta*0.1388*tan(0.5*pi*pd1.alpha);
    act_sigmas(j-1) = mean(sigmas);
    act_nus(j-1) = mean(nus);
    lags(j-1) = j;
    
    fprintf('%i \n', j)
end
%maybe say used calc_delta as real? even though doesn't seem to work...
%test value of equation fully within one distribution just to confirm

% SIGMA: do second cos relies on nu
% Lagrangian minimization for alpha
%lagrangian = @(alpha) immse(act_sigmas,pd1.sigma*lags.^(1/alpha));
%alpha = fmincon(lagrangian,1.5);

%figure()
%hold on
%plot(lags,act_sigmas)
%plot(lags,pd1.sigma*lags.^(1/alpha))

% NU
figure()
plot(lags,act_nus)
title('Nu as a function of scale size of grouped returns')
xlabel('scaling size (in days)')
ylabel('empirical value of nu')
% because it quickly goes to infinity, and as nu -> inf, tlocScale distribs
% become gaussian, we can assume scaling as a gaussian
