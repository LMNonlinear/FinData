function [ info ] = conditional_entropy2( predict, sample, given )
% Calculates conditional entropy I(X;Y|Z) = I(predict;sample|given)

% I(X;Y|Z) = I(X;Y) + H(Z,X) + H(Z,Y) + H(X,Y) - H(X) - H(Y) - H(Z) - H(X,Y,Z)

% Compute entroy of z (by same method as joint so match up)
[ds,joint_density] = joint_prob_trip([given,given,given]);
joint_density = joint_density/sum(sum(sum(joint_density*ds*ds*ds)));
Hz = -sum(sum(sum(joint_density*ds*ds*ds.*log(joint_density*ds*ds*ds))));

% Compute joint entropy of predict & given
[ds,joint_density] = joint_prob_trip([predict,given,given]);
joint_density = joint_density/sum(sum(sum(joint_density*ds*ds*ds)));
Hxz = -sum(sum(sum(joint_density*ds*ds*ds.*log(joint_density*ds*ds*ds))));

% Compute joint entropy of sample & given
[ds,joint_density] = joint_prob_trip([sample,given,given]);
joint_density = joint_density/sum(sum(sum(joint_density*ds*ds*ds)));
Hyz = -sum(sum(sum(joint_density*ds*ds*ds.*log(joint_density*ds*ds*ds))));

% Find triple here & input
[ds,joint_density] = joint_prob_trip([predict,sample,given]);
joint_density = joint_density/sum(sum(sum(joint_density*ds*ds*ds)));
Hxyz = -sum(sum(sum(joint_density*ds*ds*ds.*log(joint_density*ds*ds*ds))));

%info = mutual_info(predict,sample) + Hxy + Hxz + Hyz - Hx - Hy - Hz - Hxyz;
info =  Hxz + Hyz - Hz - Hxyz;

end