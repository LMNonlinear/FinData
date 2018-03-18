function [ CVaR ] = CVaR(pd_obj, percent)
%Calculate expected shortfalls from a given distribution object
    %Note that CVaR(i) is loss of i% or more, NOT confidence level i%

% Set percentage range to calculate vars over
x = 0.01:0.01:percent; %is this running the right way? just be sure

%Calculate VaRs from cdf
vars = icdf(pd_obj,x/100);

%Integrate over computed VaRs to find average
CVaR = sum(vars)/length(x);

end

