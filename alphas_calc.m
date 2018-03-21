function [ ] = alphas_calc( data_set, epd_object)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if min(data_set) < 0
    data_set = flip(sort(data_set));
    log_prob = log(pdf(epd_object, data_set));
    data_set = abs(data_set);
else 
    data_set = sort(data_set);
    log_prob = log(pdf(epd_object, data_set));
end
log_data = log(data_set);

params = zeros(2,length(data_set)-2);
for i = 1:(length(data_set)-2)
    new_data = log_data(end-1-i:end);
    params(:,i) = [ones(length(new_data),1), new_data]\log_prob(end-1-i:end);
end

figure()
x = 3:length(data_set);
plot(x,-params(2,:))
title('Power law decay constant vs. number of values used to compute','fontsize',14)
xlabel('number of data points used (from most extreme)','fontsize',14)
ylabel('alpha','fontsize',14)
set(gca,'Xdir','reverse')

%figure()
%hold on
%plot(log_data, log_prob, '+b')
%x = linspace(min(log_data),max(log_data),2); %2 -> 100
%y = zeros(length(params(1,:)),length(x));
%for i = 1: length(params(1,:))
%    y(i,:) = params(1,i) + params(2,i)*x;
%    plot(x,y(i,:),'-g')
%end
%title(['Fitted plot of probability vs. extreme ',set_name],'fontsize',14)
%xlabel(['log(',set_name,')'],'fontsize',14)
%ylabel('log(probability)','fontsize',14)

%alphas = params(2,:);
end

