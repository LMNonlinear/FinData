%Assignment 1: personal data
%NO LIM, USE FULL!

%Import data and divide into sections
prices = struct2array(load('pdata.mat', 'pdata'))';
t = (1:length(prices))';


%% USELESS SECTION!!
%Find an emipirical distribution for log-returns and show isn't Gaussian

%Compute returns over limited and full range of t
returns_full = diff(prices)./prices(1:end-1);

%Set number of bins for histogram according to standard Excel practice:
%http://cameron.econ.ucdavis.edu/excel/ex11histogram.html
nb1 = round(length(returns_full)^0.5);

%Pull histogram data
[counts_f, edges_f] = histcounts(returns_full,nb1);

%Modify and normalize histogram data for plotting as pdf
bin_centres_f =0.5*(edges_f(1:end-1)+edges_f(2:end));
counts_f = counts_f/(sum(counts_f)*(bin_centres_f(2)-bin_centres_f(1)));

%Plot pdfs of log returns over both limited and full range of t
figure()
hold on
plot(bin_centres_f,counts_f)

%Fit and plot a normal distribution from parameters for comparison
mu = mean(returns_full);
dev = std(returns_full);
x_norm = min(bin_centres_f):(max(bin_centres_f)-min(bin_centres_f))/ ...
    1000: max(bin_centres_f);
plot(x_norm,normpdf(x_norm,mu,dev),'-m')
legend('full set','Gaussian fit')
%bar(b1,c1/trapz(bin_centres_l,counts_l)) %optional addition of histogram

%%
%Show why we have chosen a limited range by quick visual inspection

%Observe different variation structures outside the limited range:
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
moments_defined(returns_full(200:1100))
moments_defined(returns_full)

%% PROBABLY USELESS SECTION
%Rolling distribution parameters with given window size in days
%Format as 4 subplots?

window_size = 30;
roll(returns_full(200:end),window_size)

%%
%More stationarity tests
%Should be using R_full(1:end-length(R_lim))
%0 = confirm null hypothesis
%https://stats.stackexchange.com/questions/30569/what-is-the-difference-between-a-stationary-test-and-a-unit-root-test/235916#235916
%^go to bit with cases for reference

%null hypothesis: process is trend stationary (stochastic around deterministic mean)
%alternative: process is difference stationary(stochastic mean)
%both pass @ max certainty
lag = @(set) round(sqrt(length(set)));
[decision_kp_f,p_kp_f] = kpsstest(returns_full,'lag',lag(returns_full));
[decision_kp_l,p_kp_l] = kpsstest(returns_full,'lag',lag(returns_full));

%null hypothesis: process is difference stationary
%alternative hypothesis: either the process is stationary, or trend stationary
%full fails @ min certainty, lim passes!
lag2 = @(set) 12*round((length(set))^0.25);
[decision_adf_f,p_adf_f] = adftest(returns_full,'lag',lag2(returns_full)); 
[decision_adf_l,p_adf_l] = adftest(returns_full,'lag',lag2(returns_full)); 

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
figure()
autocorr(returns_full,100)
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
scale_comparer(returns_full,'tLocationScale',7)
% Note that you constructed scaled distributions from ordered subsets, e.g.
% 1 week of consecutive days from day 1,2,3,4,5,6, instead of from random
% daily returns, because you believed order to matter after seeing effects
% on scaled alpha from perfectly ordering the ideal distribution

% Compute scaled mean out of sample errors & plot
%[V95ers,V99ers,CV95ers,CV99ers] = validator(returns_lim,'Stable');
% Ordinary log-linear fit better than adjusted alpha

% Use winner (log-linear) to make predictions of mean, var, & cvar into future
% if need this, fix dates thing
% predictor(returns_full,400,'tLocationScale',dates)
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

r_ppdf_obj = fitdist(returns_full,'tLocationScale');

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

figure()
hold on
plot(q,q.*mean_hursts','r')
plot(q,q.*mean_hursts' + q.*std_hursts'./2,'b',q,q.*mean_hursts'-q.*std_hursts'./2,'b')
%plot(q,compare_plot)
title('Multi-scaling scaling function vs. degree of distribution moment')
xlabel('degree of distribution moment (q)')
ylabel('q * H(q)')
legend({'generalized Hurst exponent','error margins on calculated exponent'},...
    'location','SouthWest')
% fix labels to represent multiplication by q

% The value of H(q) give indication about the fractal nature of the signal.
% H(q) = 0.5 corresponds to a Brownian motion, deviations form 0.5 and 
% dependency on q are indications of multi-fractality and time-correlations

%% Find non-parametric VaR and CVaR for returns

%Compute returns & appropriate empirical cdf over limited range
rank_lim = 1:length(returns_full);
ecdf_lim = rank_lim/(length(rank_lim)+1); %used rank method to generate cdf

%Compute returns & appropriate empirical cdf over full range
rank_full = 1:length(returns_full);
ecdf_full = rank_full/(length(rank_full)+1);

%Plot empirical cdf as generated by rank method (no smoothing)
figure()
hold on
ordered_ret_lim = sort(returns_full);
plot(ordered_ret_lim,ecdf_lim)

%Find a better cdf using a kernel to estimate the true empirical pdf
%Here we use epanechnikov kernel, because most efficient according to: 
%https://doi.org/10.1137/1114019
ret_epdf_obj = empirical_pdf_object(returns_full);
returns_x = min(returns_full):0.0001:max(returns_full);
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
nb = round(length(returns_full)^0.6);
[counts, edges] = histcounts(returns_full,nb);
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

%Create empirical pdf and cdf of log returns
R_epdf_object = empirical_pdf_object(returns_full);
R_x = min(returns_full):0.0001:max(returns_full);
R_epdf = pdf(R_epdf_object,R_x);
R_ecdf = cumtrapz(R_epdf);
R_ecdf = R_ecdf/max(R_ecdf);

%Make parametric pdf and cdf
R_ppdf_object = fitdist(returns_full,'tLocationScale');
R_ppdf = pdf(R_ppdf_object,R_x);
R_pcdf = cumtrapz(R_ppdf);
R_pcdf = R_pcdf/max(R_pcdf);

%Plot parametric pdf against empirical cdf
figure()
hold on
plot(R_x,R_epdf)
plot(R_x,R_ppdf)
legend({'empirical pdf','fitted Student-t pdf'},'location','NorthWest')
title('Comparison of empirically and parametrically determined pdfs')
xlabel('returns')
ylabel('probability density')

%%
%Find rolling var & cvar & backtest, run for parametric & empirical

%Set size of rolling window in days
%Decide optimal lag based on results of backtesting for diff lags
lag = 65;

[rolling_vEmp, rolling_cvEmp] = rollvarES(returns_full,lag,'Empirical');
[rolling_vSt, rolling_cvSt] = rollvarES(returns_full,lag,'tLocationScale');

%ebt_emp = esbacktest(returns_lim(lag:end),rolling_vEmp',rolling_cvEmp',...
%'VaRLevel',[0.95,0.99]);

%ebt_par = esbacktest(returns_lim(lag:end),rolling_vSt',rolling_cvSt',...
%'VaRLevel',[0.95,0.99]);

%this probably more useful but need to manually interp the data
%S_emp = summary(ebt_emp);
%S_par = summary(ebt_par);

%this seems to test normal and tLocScale as the distrib for vars on its own
%defaults 95% certainty to accept
%TestResults_emp = runtests(ebt_emp); 
%TestResults_par = runtests(ebt_par);
%%
%Make rank-frequency plots for comparison to various distributions 

%Make plot for log returns compared to center-fitted Stable distribution
%rank_freq2(R_full(R_full>0),'exponential','log returns')
rank_freq(returns_full, 'tLocationScale','returns')

%Make plot for returns and compare to Gaussian
%rank_freq(returns_full, 'normal', 'returns')

%% Find tails distribution if exist

%Isolate needed data points and take log
returns_sorted = sort(returns_full);
ret_lim_extremes_neg = returns_sorted(1:6); %last 6 used as example
ret_lim_extremes_pos = returns_sorted(end-5:end); %last 6 as example
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
returns_x2 = [linspace(min(returns_full),max(-ret_lim_extremes_neg),100)...
    linspace(min(ret_lim_extremes_pos),max(returns_full),100)];
returns_tailpdf = constant*abs(returns_x2).^slope;
figure()
hold on
plot(returns_x2,returns_tailpdf,'rx') %negatives appear to be non-power
plot(returns_x,smooth_epdf)

%% Determine how power-low alpha changes based on number of points used:

%Done for log returns
%R_epdf_object = empirical_pdf_object(returns_full);
%alphas_calc(R_lim(R_lim>0),R_epdf_object);
%alphas_calc(R_lim(R_lim<0),R_epdf_object);

%Done for returns
ret_epd_object = empirical_pdf_object(returns_full);
%alphas_calc(returns_lim(returns_lim>0),ret_epd_object);
alphas_calc(returns_full(returns_full<0),ret_epd_object);
alphas_calc(returns_full(returns_full>0),ret_epd_object);

%% Kolmogorov-Smirnov test & Anderson-Darling test

%0 = accepted, 1 = rejected, p best for Stable!
%cv = critical value between accept & reject, stat = determined value
%significance level (theta) 'Alpha' = 0.05 as default = 5% 
%null hypothesis = same distrib, rejected at level alpha
test = fitdist(returns_full,'tLocationScale');
[decision_ks,p_ks,ksstat,cv_ks] = kstest(returns_full,'CDF',test);
[decision_ad,p_ad,adstat,cv_ad] = adtest(returns_full,'Distribution',test);
%only interested in p values: compare p_ks and p_ad to see if fits centre
%better than tails

%%
%QQ plots

%Set up sample, input distribution for comparison, and compare
figure()
sample = returns_full;
PD = fitdist(sample,'tLocationScale'); 
qqplot(sample,PD)

%Heavy tailed relative to exponential (both pos & neg)
%figure()
%sample_exp = abs(returns_lim - mean(returns_lim));
%PD_exp = fitdist(sample_exp,'exponential'); 
%qqplot(sample_exp,PD_exp)

