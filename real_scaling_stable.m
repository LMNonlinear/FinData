function [ b_gam, b_delt ] = real_scaling_stable(daily_returns, dist_type)
%Fits various models to data of scaled distributions to determine best one

emp_dist1 = fitdist(daily_returns,dist_type);

runs = 49;
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
figure()
hold on
plot(scales,act_deltas)
plot(scales,b_delt(1)+b_delt(2)*scales)
plot(scales,emp_dist1.delta*scales.^(1/emp_dist1.alpha))
title('Comparing different models for the delta of a scaled returns pdf')
xlabel('log(scaling size (in days))')
ylabel('log(computed delta)')
legend({'empirical deltas','linear fit of deltas','theoretical deltas'},...
    'location','NorthWest')

% Lagrangian minimization for alpha tuning param (gam)
lagrangian = @(x) immse(log(scales)./(log(act_gams/emp_dist1.gam)), ...
    emp_dist1.alpha + x*emp_dist1.alpha*scales);
param = fmincon(lagrangian,0.1);

% gams
llag = log(scales);
lgam = log(act_gams);
loglags = [ones(1,length(llag)); llag];
b_gam = loglags'\(lgam)';
figure()
hold on
plot(llag,lgam)
plot(llag,b_gam(1)+b_gam(2)*llag)
plot(llag,log(emp_dist1.gam)+(1/emp_dist1.alpha)*llag);
plot(llag,log(emp_dist1.gam)+llag./(emp_dist1.alpha*(1+param*scales)));
title('Comparing different models for the gamma of a scaled returns pdf')
xlabel('log(scaling size (in days))')
ylabel('log(computed gamma)')
legend({'empirical gammas','log-linear fit of gammas','theoretical gammas',...
    'variable alpha model'},'location','NorthWest')
%This actually concludes throwing out theory altogether, but our best fit
%parameters aren't alphas at all, would be alphas if weren't allowed free
%intercept... check how

% Came upon curve idea because perfectly sorted returns scale according to
% multiplier where alpha = 1 ... perhaps mention curve in improvement
% section cos doesn't look like you'll be using

end

