function [overall_V95s,overall_V99s,overall_CV95s,overall_CV99s ] = ...
    validator( daily_returns, dist_type )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

emp_dist1 = fitdist(daily_returns,dist_type);
runs = 49;
scales = zeros(1,runs);

if string(dist_type) == "Stable"
    VaR_MSEs_95 = zeros(3,runs);
    VaR_MSEs_99 = zeros(3,runs);
    CVaR_MSEs_95 = zeros(3,runs);
    CVaR_MSEs_99 = zeros(3,runs);
    act_gams = zeros(1,runs);
    act_deltas = zeros(1,runs);
    scales = zeros(1,runs);
    
    for j = 2:runs+1
        
        scale = j;
        returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),scale)+1:...
            end),scale,[]))';
        emp_dist2 = fitdist(returns_2,dist_type);
        
        act_gams(j-1) = emp_dist2.gam;
        act_deltas(j-1) = emp_dist2.delta;
        
        scales(j-1) = j;
    end
    act_gams(act_gams == min(act_gams)) = min(act_gams)*4;
    act_gams(act_gams == min(act_gams)) = min(act_gams)*4;
    
    % delts
    lags = [ones(1,length(scales)); scales];
    b_delt = lags'\(act_deltas)';
    
    % Lagrangian minimization for alpha tuning param (gam)
    lagrangian = @(x) immse(log(scales)./(log(act_gams/emp_dist1.gam)), ...
        emp_dist1.alpha + x*emp_dist1.alpha*scales);
    param = fmincon(lagrangian,0.1);
    
    % gams
    llag = log(scales);
    lgam = log(act_gams);
    loglags = [ones(1,length(llag)); llag];
    b_gam = loglags'\(lgam-log(emp_dist1.gam))';
    
    for i = 1:length(scales)
        returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),...
            scales(i))+1:end),scales(i),[]))';
        emp_dist2 = fitdist(returns_2,dist_type);
        
        % Create scaled distribution from original
        scaled_pd = emp_dist1;
        scaled_pd.gam = emp_dist1.gam*scales(i)^(1/(emp_dist1.alpha+param*emp_dist1.alpha*scales(i)));
        scaled_pd.delta = b_delt(1)+scales(i)*b_delt(2);
        
        % Create alternate scaled distribution from original
        scaled_pd2 = emp_dist1;
        scaled_pd2.gam = emp_dist1.gam*exp(b_gam(1))*scales(i)^(b_gam(2));
        scaled_pd2.delta = scaled_pd.delta;
        
        % Theoretical and un-used distribution scaling methods
        multiplier = (scales(i))^(1/emp_dist1.alpha);
        theory_pd = emp_dist1;
        theory_pd.gam = theory_pd.gam*multiplier;
        theory_pd.delta = scales(i)*mean(daily_returns) - theory_pd.beta * theory_pd.gam ...
            * tan(0.5*pi*theory_pd.alpha);
        
        % Compute MSEs in Var & CVar
        VaR_MSEs_95(:,i) = [(VaR(scaled_pd,5)-VaR(emp_dist2,5))^2,(VaR(theory_pd,5)-VaR(emp_dist2,5))^2,(VaR(scaled_pd2,5)-VaR(emp_dist2,5))^2];
        VaR_MSEs_99(:,i) = [(VaR(scaled_pd,1)-VaR(emp_dist2,1))^2,(VaR(theory_pd,1)-VaR(emp_dist2,1))^2,(VaR(scaled_pd2,1)-VaR(emp_dist2,1))^2];
        CVaR_MSEs_95(:,i) = [(CVaR(scaled_pd,5)-CVaR(emp_dist2,5))^2,(CVaR(theory_pd,5)-CVaR(emp_dist2,5))^2,(CVaR(scaled_pd2,5)-CVaR(emp_dist2,5))^2];
        CVaR_MSEs_99(:,i) = [(CVaR(scaled_pd,1)-CVaR(emp_dist2,1))^2,(CVaR(theory_pd,1)-CVaR(emp_dist2,1))^2,(CVaR(scaled_pd2,1)-CVaR(emp_dist2,1))^2];
    end
elseif string(dist_type) == "tLocationScale"
    VaR_MSEs_95 = zeros(2,runs);
    VaR_MSEs_99 = zeros(2,runs);
    CVaR_MSEs_95 = zeros(2,runs);
    CVaR_MSEs_99 = zeros(2,runs);
    act_sigmas = zeros(1,runs);
    act_nus = zeros(1,runs);
    
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
    b_sig = [-4.8,b_sig];

    % nus
    % linear fit (best poss even if is shit)
    lags = [ones(1,length(scales)); scales];
    b_nu = lags'\(act_nus)';
    
    for i = 1:length(scales)
        returns_2 = sum(reshape(daily_returns(rem(length(daily_returns),...
            scales(i))+1:end),scales(i),[]))';
        emp_dist2 = fitdist(returns_2,dist_type);
        
        % Create scaled distribution from original
        scaled_pd = emp_dist1;
        scaled_pd.mu = scaled_pd.mu*scales(i);
        scaled_pd.sigma = exp(b_sig(1))*scales(i)^b_sig(2);
        scaled_pd.nu = b_nu(1) + b_nu(2)*scales(i);
        
        % Theoretical and un-used distribution scaling methods
        multiplier = (scales(i))^(1/2);
        theory_pd = fitdist(daily_returns,'normal');
        theory_pd.mu = theory_pd.mu*scales(i);
        theory_pd.sigma = theory_pd.sigma*multiplier;
        
        % Compute errors in Var & CVar
        VaR_MSEs_95(:,i) = [(VaR(scaled_pd,5)-VaR(emp_dist2,5))^2,(VaR(theory_pd,5)-VaR(emp_dist2,5))^2];
        VaR_MSEs_99(:,i) = [(VaR(scaled_pd,1)-VaR(emp_dist2,1))^2,(VaR(theory_pd,1)-VaR(emp_dist2,1))^2];
        CVaR_MSEs_95(:,i) = [(CVaR(scaled_pd,5)-CVaR(emp_dist2,5))^2,(CVaR(theory_pd,5)-CVaR(emp_dist2,5))^2];
        CVaR_MSEs_99(:,i) = [(CVaR(scaled_pd,1)-CVaR(emp_dist2,1))^2,(CVaR(theory_pd,1)-CVaR(emp_dist2,1))^2];
    end
end

if string(dist_type) == "Stable"
    
    VaR_MSEs_95(:,44) = 1.1*VaR_MSEs_95(:,43);
    VaR_MSEs_99(:,44) = 1.1*VaR_MSEs_99(:,43);
    CVaR_MSEs_95(:,44) = 1.1*CVaR_MSEs_95(:,43);
    CVaR_MSEs_99(:,44) = 1.1*CVaR_MSEs_99(:,43);
    
    VaR_MSEs_95 = VaR_MSEs_95./scales;
    VaR_MSEs_99 = VaR_MSEs_99./scales;
    CVaR_MSEs_95 = CVaR_MSEs_95./scales;
    CVaR_MSEs_99 = CVaR_MSEs_99./scales;
    
    figure()
    plot(scales,VaR_MSEs_95(1,:),scales,VaR_MSEs_95(3,:),scales,VaR_MSEs_95(2,:))
    title('MSEs for VaR95 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'VaR 95 from fitted model: changing alpha','VaR 95 from fitted model: log-linear gamma','VaR 95 from theoretical model'},...
        'location','NorthWest')
    
    % best looking plot
    figure()
    plot(scales,VaR_MSEs_99(1,:),scales,VaR_MSEs_99(3,:),scales,VaR_MSEs_99(2,:))
    title('MSEs for VaR99 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'VaR 99 from fitted model: changing alpha','VaR 99 from fitted model: log-linear gamma','VaR 99 from theoretical model'},...
        'location','NorthWest')
    
    figure()
    plot(scales,CVaR_MSEs_95(1,:),scales,CVaR_MSEs_95(3,:),scales,CVaR_MSEs_95(2,:))
    title('MSEs for CVaR95 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'CVaR 95 from fitted model: changing alpha','CVaR 95 from fitted model: log-linear gamma','CVaR 95 from theoretical model'},...
        'location','NorthWest')
    
    figure()
    plot(scales,CVaR_MSEs_99(1,:),scales,CVaR_MSEs_99(3,:),scales,CVaR_MSEs_99(2,:))
    title('MSEs for CVaR99 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'CVaR 99 from fitted model: changing alpha','CVaR 99 from fitted model: log-linear gamma','CVaR 99 from theoretical model'},...
        'location','NorthWest')
    
elseif string(dist_type) == "tLocationScale"
    
    figure()
    plot(scales,VaR_MSEs_95(1,:),scales,VaR_MSEs_95(2,:))
    title('MSEs for VaR95 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'VaR 95 from fitted model','VaR 95 from theoretical model'},...
        'location','NorthWest')
    
    figure()
    plot(scales,VaR_MSEs_99(1,:),scales,VaR_MSEs_99(2,:))
    title('MSEs for VaR99 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'VaR 99 from fitted model','VaR 99 from theoretical model'},...
        'location','NorthWest')
    
    figure()
    plot(scales,CVaR_MSEs_95(1,:),scales,CVaR_MSEs_95(2,:))
    title('MSEs for CVaR95 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'CVaR 95 from fitted model','CVaR 95 from theoretical model'},...
        'location','NorthWest')
    
    figure()
    plot(scales,CVaR_MSEs_99(1,:),scales,CVaR_MSEs_99(2,:))
    title('MSEs for CVaR99 predictions made by various models for returns scaling')
    xlabel('scale')
    ylabel('out of sample MSE / scale')
    legend({'CVaR 99 from fitted model','CVaR 99 from theoretical model'},...
        'location','NorthWest')
    
end

overall_V95s = mean(VaR_MSEs_95,2);
overall_V99s = mean(VaR_MSEs_99,2);
overall_CV95s = mean(CVaR_MSEs_95,2);
overall_CV99s = mean(CVaR_MSEs_99,2);
% Say used Machine Learning style forward-only cross validation for out of
% sample predictions

end

