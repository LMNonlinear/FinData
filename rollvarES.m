function [roll_vars, roll_cvars] = rollvarES( data_set, lag, distrib_type )
%Find rolling var & cvars at 95 & 99% confidence given set lag in days

%Create empty vectors for rolling var & cvar: top row 5%, bottom 1%
roll_vars = zeros(2,length(data_set)-lag);
roll_cvars = zeros(2,length(roll_vars)-lag);

%Assign rolling var & cvar entries: results will be from day(lag) to end
for i = 1:length(roll_vars)
    data_temp = data_set(i:i+lag-1);
    if strcmp(distrib_type,'Empirical') == 1
        ppdf_obj_temp = empirical_pdf_object(data_temp);
    else
        ppdf_obj_temp = fitdist(data_temp,distrib_type);
    end
    roll_vars(:,i) = [VaR(ppdf_obj_temp,5),VaR(ppdf_obj_temp,1)];
    roll_cvars(:,i) = [CVaR(ppdf_obj_temp,5),CVaR(ppdf_obj_temp,1)];
end
%esbacktest(R_lim,Rolling_var_data,rolling_cvar_data,'VaRLevel',0.95)

end

