function [ VaR ] = VaR( pd_obj, percent )
%Calculate values at risk from a given distribution object
    %Note that VaR(i) is loss of i%, NOT confidence level i%

%Calculate VaR from inverse cdf
VaR = icdf(pd_obj,percent/100);

end
