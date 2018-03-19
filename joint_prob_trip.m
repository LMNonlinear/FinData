function [ new_step, joint_density ] = joint_prob_trip( data )
% Input data shoud be of form [data1,data2,data3]

steps = min(min(data)):0.01:max(max(data));

%2D: 
% Generate three discrete random variables (uncorrelated)
%numStates = 2*round(sqrt(length(data(:,1))));
numStates = length(data)/3;
rescaler = numStates/(max(steps)-min(steps));
data_new = round(rescaler*(data-min(steps)));
% Count the number of (x,y,z) values in each possible state
% This is a 2D map of joint probability counts
joint_density = accumarray(data_new+1,1);
joint_density(joint_density == 0) = 0.000000001;

new_step = 1/rescaler;
%new_steps = min(steps):new_step:max(steps);

%[X,Y] = meshgrid(new_steps(1:size(joint_density,1)),new_steps(1:size(joint_density,2)));
%figure()
%surf(X,Y,joint_density')

end

