function [ ] = rank_freq2( data_set, distribution, set_name)
%For distribution, input either distribution name or a distribution object
%Generates rank-frequency plot for partial distribution

%Generate rank-frequency plot from data set
figure()
if min(data_set) > 0
    data = sort(data_set);
else
    data = sort(-data_set);
end
loglog(data,1-(1:(length(data)))/(length(data)+1),'xr')
hold on

%Create rank-frequency plot of fitted distribution:
%Set x-axis for both positive and negative data set values
x = linspace(min(data_set),max(data_set),10001);

%Create normalized pdfs for each half of the distribution
if ischar(distribution) == 1
    ppdf_object = fitdist(data_set, distribution);
else 
    ppdf_object = distribution;
    distribution = distribution.DistributionName;
end
if min(data_set) > 0
    pdf_x = pdf(ppdf_object,x);
else
    pdf_x = flip(pdf(ppdf_object,x));
end

%Compute cdfs and plot
cdf = cumsum(pdf_x)*abs(max(data_set)-min(data_set))/10000;
cdf = cdf/max(cdf);
if min(data_set) > 0
    loglog(x,1-cdf,'-m','linewidth',2)
else
    loglog(-flip(x),1-cdf,'-c','linewidth',2)
end
axis([min(data_set) 0.5 1e-3 1])
s = ' ';
legend({set_name,[distribution,' fit']},'location','NorthWest')
title(['Rank frequency plot of',s,set_name,' vs.',s,distribution,...
    ' distribution'],'fontsize',14)
xlabel(set_name,'fontsize',14)
ylabel('complemetary cumulative distribution','fontsize',14)

end

