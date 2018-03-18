function [ ] = roll( data_set, lag )
%Find rolling var & cvars at 95 & 99% confidence given set lag in days

%Create empty vectors for rolling var & cvar: top row 5%, bottom 1%
len = length(data_set)-lag;
roll_means = zeros(1,len);
roll_stds = zeros(1,len);
roll_skews = zeros(1,len);
roll_kurts = zeros(1,len);

%Assign rolling var & cvar entries: results will be from day(lag) to end
for i = 1:len
    data_temp = data_set(i:i+lag-1);
    roll_means(i) = mean(data_temp);
    roll_stds(i) = std(data_temp);
    roll_skews(i) = skewness(data_temp);
    roll_kurts(i) = kurtosis(data_temp);
end

x = 1:len;
figure()
subplot(4,1,1);
plot(x,roll_means)
title(['Rolling mean with ',num2str(lag),' day lag'])
xlabel('date')
ylabel('calculated value over window')

subplot(4,1,2);
plot(x,roll_stds)
title(['Rolling standard deviation with ',num2str(lag),' day lag'])
xlabel('date')
ylabel('calculated value over window')

subplot(4,1,3);
plot(x,roll_skews)
title(['Rolling skewness with ',num2str(lag),' day lag'])
xlabel('date')
ylabel('calculated value over window')

subplot(4,1,4);
plot(x,roll_kurts)
title(['Rolling kurtosis with ',num2str(lag),' day lag'])
xlabel('date')
ylabel('calculated value over window')

end

