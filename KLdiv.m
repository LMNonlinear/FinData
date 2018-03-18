function [KLD] = KLdiv( data1 , data2 )
% Calculates Kullback-Leibler divergence between two distributions

x = min(min(data1),min(data2)):0.0001:max(max(data1),max(data2));
dist1 = empirical_pdf_object(data1);
dist2 = empirical_pdf_object(data2);

pdf1 = pdf(dist1,x);
pdf2 = pdf(dist2,x);

pdf1(pdf1 == 0) = 0.0000000001;
pdf2(pdf2 == 0) = 0.0000000001;

KLD = sum(pdf1.*log(pdf1./pdf2))*(x(2)-x(1));

end