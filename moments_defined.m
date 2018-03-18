function [ ] = moments_defined( data_set )
%Plot standard qualities as a function of data_set size
%   Detailed explanation goes here

len = length(data_set)-5;
means = zeros(1,len);
sigmas = zeros(1,len);
skews = zeros(1,len);
kurts = zeros(1,len);

for i = 1:len
    means(i) = mean(data_set(1:i));
    sigmas(i) = std(data_set(1:i));
    skews(i) = skewness(data_set(1:i));
    kurts(i) = kurtosis(data_set(1:i));
end

x = 1:len;
figure()
hold on
%plot(x, means, x, sigmas, x, skews, x, kurts)
plot(x, means)
plot(x, sigmas)
plot(x, skews)
plot(x, kurts)
xlim([min(x),max(x)])
legend({'mean','standard deviation','skewness','kurtosis'},...
    'location','East')
title('Check of whether or not moments of the distribution are defined')
xlabel('number of data points used to calculate parameter')
ylabel('calculated value of parameter')

end

