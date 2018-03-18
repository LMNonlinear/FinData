%create pure stable data for testing theory
%maybe use this to show errors on model by bootstrapping to find standard
%variation in alpha, beta, etc. and show that your predicitions lie within
%this error
%% Bit from scale_comparer to set up

lag1 = 1;

%returns spaced by lag1

%lagged2 returns
%returns_2 = sum(reshape(returns_1(rem(length(returns_1),lag2)+1:end),lag2,[]))';

% Create comparison distributions from data
%emp_dist2 = fitdist(returns_2,dist_type);

%% Ideal testing

x = 0.0001:0.0001:1-0.0001;
pd1 = fitdist(daily_returns,dist_type);

% This is ideal starting data
y = icdf(pd1,x)';

lag2 = 2;
calc_deltas = zeros(1,100);
act_deltas = zeros(1,100);

for j = 2:102
    
    lag2 = j;
    gams = zeros(1,100);
    delts = zeros(1,100);
    means = zeros(1,100);
    for i = 1:100
        y = y(randperm(length(y)));
        
        % Ideal scaled data
        y2 = sum(reshape(y(rem(length(y),lag2)+1:end),lag2,[]))';
        pd2 = fitdist(y2,dist_type);
        gams(i) = pd2.gam;
        delts(i) = pd2.delta;
        means(i) = mean(y2);
    end
    
    calc_deltas(j) = mean(means)-pd1.beta*0.1388*tan(0.5*pi*pd1.alpha);
    act_deltas(j) = mean(delts);
    
    fprintf('%i \n', j)
end
%maybe say used calc_delta as real? even though doesn't seem to work...
%test value of equation fully within one distribution just to confirm

calc = [ones(1,length(calc_deltas)); calc_deltas];
b = calc'\act_deltas';
figure()
hold on
plot(calc_deltas,act_deltas)
plot(calc_deltas,b(1)+b(2)*calc_deltas)
plot(calc_deltas,calc_deltas)

xplot = min(returns_2):0.01:max(returns_2);

figure()
hold on
plot(xplot,pdf(pd1,xplot))
plot(xplot,pdf(pd2,xplot))
plot(xplot,pdf(emp_dist2,xplot),'ro')
