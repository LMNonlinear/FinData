function [ info ] = conditional_entropy( predict, sample, given )
% Calculates conditional entropy I(X;Y|Z) = I(predict;sample|given)

% I(X;Y|Z) = I(X;Y) + H(Z,X) + H(Z,Y) + H(X,Y) - H(X) - H(Y) - H(Z) - H(X,Y,Z)

%  NORMALIZE SELF-RESULTS BETWEEN DENSITY FUNCTIONS!!!!

% Compute entroy of x (by same method as joint so match up)
[~,density,X,Y] = joint_prob([predict,predict]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

density = density/sum(sum(density*dx*dy));
Hx = -sum(sum(density*dx*dy.*log(density*dx*dy)));

% Compute entroy of y (by same method as joint so match up)
[~,density,X,Y] = joint_prob([sample,sample]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

density = density/sum(sum(density*dx*dy));
Hy = -sum(sum(density*dx*dy.*log(density*dx*dy)));

% Compute entroy of z (by same method as joint so match up)
[~,density,X,Y] = joint_prob([given,given]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

density = density/sum(sum(density*dx*dy));
Hz = -sum(sum(density*dx*dy.*log(density*dx*dy)));

% Compute joint entropy of predict & sample
[~,joint_density,X,Y] = joint_prob([predict,sample]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

joint_density = joint_density/sum(sum(joint_density*dx*dy));
Hxy = -sum(sum(joint_density*dx*dy.*log(joint_density*dx*dy)));

% Compute joint entropy of predict & given
[~,joint_density,X,Y] = joint_prob([predict,given]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

joint_density = joint_density/sum(sum(joint_density*dx*dy));
Hxz = -sum(sum(joint_density*dx*dy.*log(joint_density*dx*dy)));

% Compute joint entropy of sample & given
[~,joint_density,X,Y] = joint_prob([sample,given]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

joint_density = joint_density/sum(sum(joint_density*dx*dy));
Hyz = -sum(sum(joint_density*dx*dy.*log(joint_density*dx*dy)));

% Find triple here & input :) (remember normalizing)
[ds,joint_density] = joint_prob_trip([predict,sample,given]);
joint_density = joint_density/sum(sum(sum(joint_density*ds*ds*ds)));
Hxyz = -sum(sum(sum(joint_density*ds*ds*ds.*log(joint_density*ds*ds*ds))));
% re-normalize so get's same output as other version
Hxyz = Hxyz * 1.0149;

%info = mutual_info(predict,sample) + Hxy + Hxz + Hyz - Hx - Hy - Hz - Hxyz;
info =  Hxz + Hyz - Hz - Hxyz;

end

