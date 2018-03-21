function [ ] = rank_freq( data_set, distribution, set_name)
%For distribution, input either distribution name or a distribution object
%Generates rank-frequency plot

%Generate rank-frequency plot from data set
figure()
loglog(sort(data_set(data_set>0)),1-(1:(length(data_set(data_set>0))))/...
    (length(data_set(data_set>0)+1)),'xr')
hold on
loglog(sort(-data_set(data_set<0)),1-(1:(length(data_set(data_set<0))))/...
    (length(data_set(data_set<0)+1)),'+b')

%Create rank-frequency plot of fitted distribution:
%Set x-axis for both positive and negative data set values
x_pos = linspace(0,max(data_set),10001);
x_neg = linspace(min(data_set),0,10001);

%Set x-axes for re-normalizing distribution function
x_pos_norm = linspace(-max(data_set),max(data_set),20001);
x_neg_norm = linspace(min(data_set),-min(data_set),20001);

%Create normalized pdfs for each half of the distribution
if ischar(distribution) == 1
    ppdf_object = fitdist(data_set, distribution);
else 
    ppdf_object = distribution;
    distribution = distribution.DistributionName;
end
pdf_pos = pdf(ppdf_object,x_pos);
pdf_neg = flip(pdf(ppdf_object,x_neg));
pdf_pos_norm = pdf(ppdf_object,x_pos_norm);
pdf_neg_norm = pdf(ppdf_object,x_neg_norm);
pdf_pos = pdf_pos*sum(pdf_pos_norm)/sum(pdf_pos); %normalizing part
pdf_neg = pdf_neg*sum(pdf_neg_norm)/sum(pdf_neg);

%Compute cdfs and plot
cdf_pos = cumsum(pdf_pos)*(max(data_set))/10000;
cdf_neg = cumsum(pdf_neg)*(-min(data_set))/10000;
loglog(x_pos,1-cdf_pos,'-m','linewidth',2)
loglog(-flip(x_neg),1-cdf_neg,'-c','linewidth',2)
axis([1e-5 1 1e-4 1]) %6e-3
s = ' ';
if min(data_set) < 0
legend({['positive',s,set_name],['negative',s,set_name],[distribution,...
    ' positive fit'],[distribution,' negative fit']},'location','NorthWest')
else 
legend({set_name,[distribution,' fit']},'location','NorthWest')
end
title(['Rank frequency plot of',s,set_name,' vs.',s,distribution,...
    ' distribution'],'fontsize',14)
xlabel(set_name,'fontsize',14)
ylabel('complemetary cumulative distribution','fontsize',14)

end

