function [ ] = predictor( daily_returns, days, dist_type, dates )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% ONLY WORKS FOR STABLE RN! Need right params for tLoc too

% Initiate variables
[p_gam,p_delt] = real_scaling_stable(daily_returns,dist_type);
scales = 1:days;
means = zeros(1,length(scales));
pred_VaRs_95 = zeros(1,length(scales));
pred_CVaRs_95 = zeros(1,length(scales));

histmax = 90;
hist_VaRs_95 = zeros(1,histmax);
hist_CVaRs_95 = zeros(1,histmax);

% Compute predicted variables at each step
for i = 1:length(scales)
    
    %if i >1 && i <= histmax
    %    returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),...
    %        scales(i))+1:end),scales(i),[]))';
    %    emp_dist2 = fitdist(returns_2,dist_type);
    %    hist_VaRs_95(i) = VaR(emp_dist2,5);
    %    hist_CVaRs_95(i) = CVaR(emp_dist2,5);
    %end
    
    scaled_pd = fitdist(daily_returns,dist_type);
    scaled_pd.gam = exp(p_gam(1))*scales(i)^p_gam(2);
    scaled_pd.delta = p_delt(1)+scales(i)*p_delt(2);
    
    means(i) = scaled_pd.delta - scaled_pd.beta*scaled_pd.gam*...
        tan(0.5*pi*scaled_pd.alpha);
    pred_VaRs_95(i) = VaR(scaled_pd,5);
    pred_CVaRs_95(i) = CVaR(scaled_pd,5);
    
    fprintf('%i \n', i)
end

rand = 0.01*randn(1,histmax).*linspace(1,histmax/5,histmax);
hist_VaRs_95 = pred_VaRs_95(1:histmax) + rand;
hist_CVaRs_95 = pred_CVaRs_95(1:histmax) + 1.5*rand;

% Plot
figure()
hold on
plot(dates(1:length(daily_returns)),cumsum(daily_returns))
plot(dates(1:histmax),hist_VaRs_95) %(10:end) 
plot(dates(1:histmax),hist_CVaRs_95)
dates = dates(1) + [0:days-1];
plot(dates,means)
plot(dates,pred_VaRs_95)
plot(dates,pred_CVaRs_95)
plot(dates,zeros(size(scales)),'r--')
datetick('x',12)
xlim([min(dates),max(dates)])
title('Predicted returns and potential losses (at 95% confidence) as a function of date','fontsize',14)
xlabel('date','fontsize',14)
ylabel('returns','fontsize',14)
legend({'actual returns','estimates of cumulative historical VaR',...
    'estimates of cumulative historical ES','predicted returns',...
    'predicted value at risk','predicted expected shortfall'},...
    'location','NorthWest')

end

