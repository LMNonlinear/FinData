function [ epdf_object ] = empirical_pdf_object( data_set )
%Create empirical pdf object of data set

epdf_object = fitdist(data_set,'Kernel','Kernel','epanechnikov');

end

