function [ ] = scale_comparer( daily_returns, dist_type, scale )
% Compare empirically scaled distribution to ones obtained by various
% scaling methods. Input daily returns, distribution type, scale.

% Establish scaled returns and plotting axis
returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),scale)+1:end),scale,[]))';
x = 1.1*min(returns_2):0.001:1.1*max(returns_2);

% Create comparison distributions from returns data
emp_dist1 = fitdist(daily_returns,dist_type);
emp_dist2 = fitdist(returns_2,dist_type);

% mu = mean (scale as scale*emp_dist1.mu)

if string(dist_type) == "Stable"
    % Create scaled distribution from original
    [params_gam,params_delt] = real_scaling_stable(daily_returns,dist_type);
    scaled_pd = emp_dist1;
    scaled_pd.gam = exp(params_gam(1))*scale^params_gam(2);
    %scaled_pd.gam = emp_dist1.gam*scale^(1/(emp_dist1.alpha+param_gam*emp_dist1.alpha*scale));
    scaled_pd.delta = params_delt(1)+scale*params_delt(2);
    
    % Theoretical and un-used distribution scaling methods
    multiplier = (scale)^(1/emp_dist1.alpha);
    theory_pd = emp_dist1;
    theory_pd.gam = theory_pd.gam*multiplier;
    theory_pd.delta = scale*mean(daily_returns) - theory_pd.beta * theory_pd.gam ...
        * tan(0.5*pi*theory_pd.alpha);
    %theory_pd.delta = -0.0572 + 0.7236*theory_pd.delta; %linear fit adjust
    %(ideally multiplying data by a multiplies gamma & delta by a)
elseif string(dist_type) == "tLocationScale"
    % Create scaled distribution from original
    [params_sig,params_nu] = real_scaling_tLoc(daily_returns,dist_type);
    scaled_pd = emp_dist1;
    scaled_pd.mu = scaled_pd.mu*scale;
    scaled_pd.sigma = exp(params_sig(1))*scale^params_sig(2);
    scaled_pd.nu = params_nu(1) + params_nu(2)*scale;
    
    % Theoretical and un-used distribution scaling methods
    multiplier = (scale)^(1/2);
    theory_pd = fitdist(daily_returns,'normal');
    theory_pd.mu = theory_pd.mu*scale;
    theory_pd.sigma = theory_pd.sigma*multiplier;
end

figure()
hold on
plot(x,pdf(emp_dist2,x),'r')
plot(x,pdf(scaled_pd,x),'b')
plot(x,pdf(theory_pd,x),'g')
%xlim([min(x),max(x)])
title('Comparison of models for scaled returns pdf to empirically determined pdf')
xlabel('returns')
ylabel('probability density')
if string(dist_type) == "Stable"
    legend({'empirical scaled pdf','pdf based on fitted behaviour',...
        'theoretical scaled pdf'},'location','NorthEast')
elseif string(dist_type) == "tLocationScale"
    legend({'empirical scaled pdf','pdf based on fitted behaviour',...
        'theoretical scaled pdf'},'location','NorthWest')
end

%deviations of empirical distribution parameters from those calculated are
%likely due to grouping idiosyncracies of returns -- not fully random,
%often many high returns/low returns in same group

end

