%Assignment 2: Dash data

%% Import and set up data
%Import data and divide into sections
[numbers, text, ~] = xlsread('cryptocurrency.xlsx','Dash'); 
open.full = flip(numbers(:,1));
high.full = flip(numbers(:,2));
low.full = flip(numbers(:,3));
closep.full = flip(numbers(:,4));
volume.full = flip(numbers(:,5));
markcap.full = flip(numbers(:,6));
t = (1:length(open.full))'; %1 = Feb 14, 2014

% Do dates right
dates1 = flip(text(2:end,1));
dates = datenum(dates1);
dates_lim = dates(1055:end);

%Set range for most examinations (look only at relevant behaviours)
range = t(1055:end);

%Make data.lim be each set over the limited range
open = limit(open, range);
high = limit(high, range);
low = limit(low, range);
closep = limit(closep, range);
volume = limit(volume, range);
markcap = limit(markcap, range);

%Set modified range for plotting returns, which plot diff. betwen prices
mod_range = range(1:end-1); 

%You are asked to investigate the scaling properties of your time-series. 
%Explore how the statistical properties of financial returns change at 
%different time scales (e.g. from daily, to weekly, to monthly). Look at 
%the scaling laws of different assets and discuss the consequences for 
%portfolio formation.

%1. stationarity; deviations from random walk (e.g., autocorrelations); 
%scaling of the return distribution; tail exponent, Hurst exponent, and 
%their relationship; modeling, forecasting and validation

%% Compute full range of returns and log returns

%Compute log-returns over limited and full range of t
R_lim = log(closep.lim(2:end)) - log(closep.lim(1:end-1)); %1 day lag
R_full = log(closep.full(2:end)) - log(closep.full(1:end-1)); %1 day lag

%Compute returns over limited and full range of t
returns_lim = diff(closep.lim)./closep.lim(1:end-1);
returns_full = diff(closep.full)./closep.full(1:end-1);

%% Check for obvious periodic changes in behaviour in price

figure()
plot(t(800:end),closep.full(800:end))
title('Closing price of stock vs. time')
xlim([min(t(800:end)),max(t)])
xlabel('date')
ylabel('closing price')

figure()
plot(dates(800:end),closep.full(800:end))
title('Closing price of stock vs. time')
datetick('x',12)
xlim([min(dates(800:end)),max(dates)])
xlabel('date')
ylabel('closing price')

%% Check for log-returns distribution being Gaussian

R_epdf_obj = fitdist(R_lim,'Kernel','Kernel','epanechnikov');
R_normal_ppdf_obj = fitdist(R_lim,'Normal');
R_x = min(R_lim):0.0001:max(R_lim);
epdf = pdf(R_epdf_obj,R_x);
norm_ppdf = pdf(R_normal_ppdf_obj,R_x);

% Plot
plot(R_x,epdf,R_x,norm_ppdf)
legend({'empirical pdf','fitted Gaussian pdf'},'location','NorthEast')
title('Comparison of empirical and Gassian-fitted pdfs')
xlabel('log returns')
ylabel('probability density')

% Hit with KS & AD

%% Show why we have chosen a limited range by quick visual inspection

%Observe different variation structures outside the limited range:
figure()
plot(mod_range,returns_lim)
figure()
plot(t(2:end),returns_full)

%Kurtosis seems to be undefined before break, defined after
%Note matlab calcs skewness & kurtosis as standardized skewness & kurtosis

%tLocScale on R_lim or returns_lim gives nu ~ 2.6, theory says kurtosis
%tends to infinity for 2 <nu < 4, and skew not defined unless nu > 3, in
%which case skew = 0... therefore tLocScale definitely not appropriate 
%(can excuse skew cos 2.6 ~ 3, but not kurtosis being defined!)

%Stable on R_lim or returns_lim gives alpha = 1.5, theory says moments of 
%order alpha or higher (alpha never > 2) don't exist... 

moments_defined(returns_full)
moments_defined(returns_full(1:1100))
moments_defined(returns_lim)

%% Rolling distribution parameters with given window size in days
%Format as 4 subplots?
%Make plot full screen to see properly

window_size = 30;
roll(returns_full(200:end),window_size)

%% More stationarity tests

% 0 = confirm null hypothesis
% https://stats.stackexchange.com/questions/30569/what-is-the-difference-between-a-stationary-test-and-a-unit-root-test/235916#235916
% ^go to bit with cases for reference

% p = what threshold value would have to be to change decision
% maximum (0.10) or minimum (0.01) p-values.
% default 0.05 alpha

% P-values are the probability of obtaining an effect at least as extreme 
% as the one in your sample data, assuming the truth of the null hypothesis
% so high p value is good

%null hypothesis: process is trend stationary (stochastic around deterministic mean)
%trend stationary includes full stationary, so turn to autocorr to pick.
%alternative: process is difference stationary(stochastic mean)
%both pass @ max certainty
lag = @(set) round(sqrt(length(set)));
[decision_kp_f,p_kp_f] = kpsstest(returns_full,'lag',lag(returns_full));
[decision_kp_l,p_kp_l] = kpsstest(returns_lim,'lag',lag(returns_lim));

%null hypothesis: process is difference stationary
%alternative hypothesis: either the process is stationary, or trend stationary
%full fails @ min certainty, lim passes!
lag2 = @(set) 12*round((length(set)/100)^0.25);
[decision_adf_f,p_adf_f] = adftest(returns_full,'lag',lag2(returns_full)); 
[decision_adf_l,p_adf_l] = adftest(returns_lim,'lag',lag2(returns_lim)); 

% If pass both, raise alpha until one fails
% Auto-covaraince 'lag' part needs work to understand
% https://faculty.washington.edu/ezivot/econ584/notes/unitrootLecture2.pdf
% ^how to pick lag

% Lag here speaks to autocorrelation behaviour examined for that lag
% Maybe say you tried a couple of different lags according to different
% papers?

% Say both halves passed kpss & failed adf, full failed failed both

%% Autocorrelation
% Non-stationary series: if a time series contains a trend, then 
% correlation values will not go to zero except for very large lags 

% Short-term correlation: stationary series (not ours) often exhibit 
% short-term correlation charcterized by a fairly large value followed by 
% 2 or 3 more coefficents which, while significantly greater than zero, 
% tend to get successively smaller

% http://staff.bath.ac.uk/hssjrh/TYPED%20Lecture%2010%20Trend%20Stationarity.pdf

% Determine autocorrelation plot to see if distributions are full
% stationary or trend stationary with a trend
returns_first = returns_full(2:end-length(returns_lim));
figure()
autocorr(returns_first,100)
figure()
autocorr(returns_lim,100)
% we see no autocorrelation for both first & lim, which implies full 
% stationarity of both -> so we pick full stationarity as resultant 
% manifestation of trend stationarity either side of break

% full stationarity of first half is okay, because undefined parameters are
% same over any given lag - kurtosis over 30 days = over another 30 days,
% even if =/= over 60 days

% numLags gives xmax of plot in lags
% choice of 100 means seeing 5 points outside bounds still consistent with
% 95% confidence

% blue lines are error bounds for error in estimations -> being inside
% these means likely not statistically significant
% The default (numSTD = 2) corresponds to approximate 95% confidence bounds

%% Scaling

% Plot theoretical vs. fitted scaling behaviour relative to actual
scale_comparer(returns_lim,'Stable',7)
% Note that you constructed scaled distributions from ordered subsets, e.g.
% 1 week of consecutive days from day 1,2,3,4,5,6, instead of from random
% daily returns, because you believed order to matter after seeing effects
% on scaled alpha from perfectly ordering the ideal distribution

% Compute scaled mean out of sample errors & plot
%[V95ers,V99ers,CV95ers,CV99ers] = validator(returns_lim,'Stable');
% Ordinary log-linear fit better than adjusted alpha

% Use winner (log-linear) to make predictions of mean, var, & cvar into future
%predictor(returns_lim,450,'Stable',dates_lim)
predictor(returns_lim,400,'Stable',dates_lim)
% Plot against actual returns (should line up with means)

% determine if self-affine, uni-scaling

% "However, the relation between fractal dimension and Hurst exponent is 
% not satisfied in many real-world processes. This is a strong indication 
% that the kind of observed scaling in financial data is not simply a fractal"

% Different moments scale with different scaling laws, which we found to be
% true!

% We are dealing with a multiscaling process

% Our "theory" lines are all scaling for random walk!
% Make "theory" scaling predicted by hurst exponent/function


%% Hurst exponents

%(stable fit optional, can input what you like)
r_ppdf_obj = fitdist(returns_lim,'Stable');

% use genhurst, made by the tomato himself!
% second entry (q) is the moment number, i.e. 1=mean, 2=var, etc.
% maxT is maximum scaling window size used: we're going for 100
q = 0:0.05:4;
[mean_hursts,std_hursts]=genhurst(returns_full,q,100); 
% a process is self-affine if it scales into the same distrib w/ diff
% parameters
% a process is uni-scaling if H does not depend on q
% H < 0.5 (which we see everywhere here) means anti-persistent process (low
% memory) -> consistent with autocorellations dying very fast
% q*H(q)-1 is a "scaling function" of the multi-scaling process
% both of our H(q)s go negative, which approaches white noise where H = -1/2
% this is consistent with the autocorrelation functions showing no
% correlation, which would give a uniform power spectral density =
% definition of white noise!!

compare_plot = zeros(size(q));
for i = 1:length(q)
    if q(i) >= r_ppdf_obj.alpha+1
        compare_plot(i) = q(i).^(-1);
    else
        compare_plot(i) = 1/(r_ppdf_obj.alpha+1);
    end
end

compare_plot = zeros(size(q));
for i = 1:length(q)
    if q(i) >= r_ppdf_obj.alpha
        compare_plot(i) = r_ppdf_obj.alpha/(2*q(i));
    else
        compare_plot(i) = 0.5;
    end
end

figure()
hold on
plot(q,q.*mean_hursts','r')
plot(q,q.*mean_hursts' + q.*std_hursts'./2,'b',q,q.*mean_hursts'-q.*std_hursts'./2,'b')
%plot(q,compare_plot)
title('Multi-scaling scaling function vs. degree of distribution moment')
xlabel('degree of distribution moment (q)')
ylabel('q * H(q)')
legend({'generalized Hurst exponent','error margins on calculated exponent'},...
    'location','NorthWest')
% fix labels to represent multiplication by q

% The value of H(q) give indication about the fractal nature of the signal.
% H(q) = 0.5 corresponds to a Brownian motion, deviations form 0.5 and 
% dependency on q are indications of multi-fractality and time-correlations

%% Find non-parametric VaR and CVaR for returns

%Compute returns & appropriate empirical cdf over limited range
rank_lim = 1:length(returns_lim);
ecdf_lim = rank_lim/(length(rank_lim)+1); %used rank method to generate cdf

%Compute returns & appropriate empirical cdf over full range
rank_full = 1:length(returns_full);
ecdf_full = rank_full/(length(rank_full)+1);

%Plot empirical cdf as generated by rank method (no smoothing)
figure()
hold on
ordered_ret_lim = sort(returns_lim);
plot(ordered_ret_lim,ecdf_lim)

%Find a better cdf using a kernel to estimate the true empirical pdf
%Here we use epanechnikov kernel, because most efficient according to: 
%https://doi.org/10.1137/1114019
ret_epdf_obj = empirical_pdf_object(returns_lim);
returns_x = min(returns_lim):0.0001:max(returns_lim);
smooth_epdf = pdf(ret_epdf_obj,returns_x);
smooth_ecdf = cdf(ret_epdf_obj,returns_x);

%Plot smoothed empirical cdf
plot(returns_x,smooth_ecdf)
legend('rank method cdf','kernel smoothed cdf')

%Create a list of VaRs & CVaRs where vars(i) = VaR(i percent)
emp_vars = zeros(1,99);
emp_cvars = zeros(1,99);
for i = 1:99
    emp_vars(i) = VaR(ret_epdf_obj,i);
    emp_cvars(i) = CVaR(ret_epdf_obj,i);
end

%Plot returns pdf & histogram with vertical lines indicating VaRs & CVaRs
figure()
hold on
plot(returns_x,smooth_epdf)

%Create returns histogram
nb = round(length(returns_lim)^0.6);
[counts, edges] = histcounts(returns_lim,nb);
bin_centres =0.5*(edges(1:end-1)+edges(2:end));
counts = counts/(sum(counts)*(bin_centres(2)-bin_centres(1)));

%Plot returns histogram
bar(bin_centres,counts/trapz(bin_centres,counts))

%Plot vertical lines corresponding to the values of chosen VaR & CVaRs
%v1 = vline1(vars(10),'r','Var10');
%v2 = vline2(cvars(10),'g','CVar10');
v3 = vline1(emp_vars(5),'r','Var5');
v4 = vline2(emp_cvars(5),'g','CVar5');

%% Find parametric VaR & CVaR for returns

% Generating a set of vars & cvars at all percentage losses
% For percentage confidence, flip results
par_vars = zeros(1,99);
par_cvars = zeros(1,99);
for i = 1:99
    par_vars(i) = VaR(r_ppdf_obj,i);
    par_cvars(i) = CVaR(r_ppdf_obj,i);
end

%%
%Create emirical pdf & cdf of log returns, & fit center of distribution

%Make parametric pdf and cdf
r_ppdf_obj.gam = 0.0384010; %adjust fit so that fits center better 
r_ppdf = pdf(r_ppdf_obj,returns_x);
r_pcdf = cdf(r_ppdf_obj,returns_x);

%Plot parametric pdf against empirical cdf
figure()
hold on
plot(returns_x,smooth_epdf)
plot(returns_x,r_ppdf)
legend({'empirical pdf','fitted Stable pdf'},'location','NorthEast')
title('Comparison of empirically and parametrically determined pdfs')
xlabel('returns')
ylabel('probability density')

%Reset r_ppdf to non-fake properties
r_ppdf_obj = fitdist(returns_lim,'Stable');

%% Find rolling VaR & CVaR & backtest 
% (run for parametric & empirical)
% Still slow but much faster -- probably 5-10 mins to run Stable due to
% convergence problems (empirical instant)

%https://uk.mathworks.com/help/risk/value-at-risk-estimation-and-backtesting-1.html
%^Var backtest help

%Set size of rolling window in days
%Decide optimal lag based on results of backtesting for diff lags
lag = 65;

[rolling_vEmp, rolling_cvEmp] = rollvarES(returns_lim,lag,'Empirical');
%[rolling_vSt, rolling_cvSt] = rollvarES(returns_lim,lag,'Stable');

%ebt_emp = esbacktest(returns_lim(lag:end),rolling_vEmp',rolling_cvEmp',...
%'VaRLevel',[0.95,0.99]);
% varbacktest is also a thing! (similar)

%ebt_par = esbacktest(returns_lim(lag:end),rolling_vSt',rolling_cvSt',...
%'VaRLevel',[0.95,0.99]);

%this probably more useful but need to manually interp the data
%S_emp = summary(ebt_emp);
%S_par = summary(ebt_par);

%this seems to test normal and tLocScale as the distrib for vars on its own
%defaults 95% certainty to accept
%TestResults_emp = runtests(ebt_emp); 
%TestResults_par = runtests(ebt_par);

%% Make rank-frequency plots for comparison to various distributions 

%Make plot for returns compared to center-fitted Stable distribution
%rank_freq2(returns_full(returns_full>0),'exponential','returns')
rank_freq(returns_lim, 'Stable','returns')

%Make plot for returns and compare to Gaussian
%rank_freq(returns_lim, 'normal', 'returns')

%The fact that points dip under rank-frequency plot suggests that alpha
%value of stable distribution is too low for tails (more
%exponential/gaussian than ppdf_obj.alpha suggests)

%Note that "power law alpha" = 1 + "stable alpha"

%% Find tails distribution if exist
% Unecessary section, they don't!

%Isolate needed data points and take log
ret_lim_extremes_neg = returns_lim(1:6); %last 6 used as example
ret_lim_extremes_pos = returns_lim(end-5:end); %last 6 as example
log_negs = log(ret_lim_extremes_neg);
log_poss = log(ret_lim_extremes_pos);

%Determine log of corresponding probabilities and fit with power law decays
log_neg_prob = log(pdf(ret_epdf_obj,ret_lim_extremes_neg));
log_pos_prob = log(pdf(ret_epdf_obj,ret_lim_extremes_pos));
params_neg = [ones(length(log_negs),1) log_negs]\log_neg_prob;
params_pos = [ones(length(log_poss),1) log_poss]\log_pos_prob;

%Plot
figure()
hold on
plot(log_negs, log_neg_prob, '+b')
x_neg = linspace(min(log_negs),max(log_negs),1000);
y_neg = params_neg(1) + params_neg(2)*x_neg;
plot(x_neg,y_neg,'-c')
legend({'extreme values','linear fit'},'location','NorthEast')
title('Fitted plot of probability vs. extreme negative returns','fontsize',14)
xlabel('log(returns)','fontsize',14)
ylabel('log(probability)','fontsize',14)

figure()
hold on
plot(log_poss, log_pos_prob,'xr')
x_pos = linspace(min(log_poss),max(log_poss),1000);
y_pos = params_pos(1) + params_pos(2)*x_pos;
plot(x_pos,y_pos,'-m')
legend({'extreme values','linear fit'},'location','NorthEast')
title('Fitted plot of probability vs. extreme positive returns','fontsize',14)
xlabel('log(returns)','fontsize',14)
ylabel('log(probability)','fontsize',14)

%Convert the results from log scale back to regular and plot
intercepts = [params_pos(1),params_neg(1)];
intercept = mean(intercepts);
constant = exp(intercept);
slopes = [params_pos(2),params_neg(2)];
slope = mean(slopes);
returns_x2 = [linspace(min(returns_lim),max(-ret_lim_extremes_neg),100)...
    linspace(min(ret_lim_extremes_pos),max(returns_lim),100)];
returns_tailpdf = constant*abs(returns_x2).^slope;
figure()
hold on
plot(returns_x2,returns_tailpdf,'rx') %negatives appear to be non-power
plot(returns_x,smooth_epdf)

%% Determine how power-law alpha changes based on number of points used

alphas_calc(returns_lim(returns_lim<0),ret_epdf_obj);
alphas_calc(returns_lim(returns_lim>0),ret_epdf_obj);

%% Kolmogorov-Smirnov test & Anderson-Darling tests

%0 = accepted, 1 = rejected, p best for Stable!
%cv = critical value between accept & reject, stat = determined value
%significance level (theta) 'Alpha' = 0.05 as default = 5% 
%null hypothesis = same distrib, rejected at level alpha

test = fitdist(returns_lim,'Stable');
[decision_ks,p_ks,ksstat,cv_ks] = kstest(returns_lim,'CDF',test);
[decision_ad,p_ad,adstat,cv_ad] = adtest(returns_lim,'Distribution',test);

%only interested in p values: compare p_ks and p_ad to see if fits centre
%better than tails

%% QQ plots

%Set up sample, input distribution for comparison, and compare
figure()
sample = abs(returns_lim - mean(returns_lim));
sample = sample(sample>0.1);
PD = fitdist(sample,'GeneralizedPareto'); 
qqplot(sample,PD)

%Heavy tailed relative to exponential (both pos & neg)
%figure()
%sample_exp = abs(returns_lim - mean(returns_lim));
%PD_exp = fitdist(sample_exp,'exponential'); 
%qqplot(sample_exp,PD_exp)


%% Thoughts

%Should be checking all quantities for both rolling windows AND lagged data
%sets: e.g. for rolling 1 year window of daily data, and rolling 1 year
%window of weekly data. Do this because non-daily data has lower noise and
%complexity, so if produces nearly as good results as daily data (shouldn't
%ever be as good?), then probably a better method. Do window because older
%information could be non-relevant clutter.
% -> find best time horizon and sampling set

%Work on backtesting function and interpretation. Find a way to work with 
%distributions if a tail does exist! (e.g. when get to diff cryptos)
%^hold of on tail bit until hit crypto2, may not be necessary.

%What should happen (don't bother writing some of code say you do):
%For all time scales (daily vs. weekly...), stationarity should be 
%consistent either side of the breaking point found last time, and break
%across it. Can code to look in more detail 

%Is stationarity cycle well placed? Shouldn't we re-test for log-normal
%distributions if have to break up data? (fix method!)

%Do bootstrapping somewhere?

%Stable distribution fit with alpha<2 (non-gaussian) implies roughly power
%law behaviour of tails; however, this behaviour is never perfectly power
%law, so fitting separate power law tails is hard: excellent graph of
%power law behaviour for diff alpha found in "Special cases" section of:
%https://en.wikipedia.org/wiki/Stable_distribution
%is there much point in fitting separate power law tails if you have fitted
%a stable? not reall, except for curiosity of what decay constant would
%be-ish