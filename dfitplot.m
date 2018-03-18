function [ object ] = dfitplot( data_set, distrib_type, set_name )
%Fit desired distribution to data and plot against empirical distribution

%Make pdf objects
epdf_object = empirical_pdf_object(data_set);
ppdf_object = fitdist(data_set,distrib_type);

%Make pdfs
x = min(data_set):0.0001:max(data_set);
ppdf = pdf(ppdf_object,x);
epdf = pdf(epdf_object,x);

%Plot
figure()
plot(x,epdf,x,ppdf)
legend({'empirical pdf','parametric pdf'},'location','NorthEast')
title('Comparison of empirically and parametrically determined pdfs')
xlabel(set_name)
ylabel('probability density')

object = ppdf_object;

end

%dfitplot(R_lim,'Stable','log returns')
