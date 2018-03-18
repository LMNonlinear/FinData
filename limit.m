function [ data ] = limit( data, range )
%This function simply cuts the full data structure to be over the range

data.lim = data.full(range);

end

