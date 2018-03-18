function [info] = mutual_info( data1 , data2 )
% Calculates mutual information between two distributions

% I = H(X) + H(Y) - H(X,Y)

% Compute entroy of x (by same method as joint so match up)
[~,density,X,Y] = joint_prob([data1,data1]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

density = density/sum(sum(density*dx*dy));
Hx = -sum(sum(density*dx*dy.*log(density*dx*dy)));

% Compute entroy of y (by same method as joint so match up)
[~,density,X,Y] = joint_prob([data2,data2]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

density = density/sum(sum(density*dx*dy));
Hy = -sum(sum(density*dx*dy.*log(density*dx*dy)));

% Compute joint entropy (I think computed based on bivariate normal
% assumptions)
[~,joint_density,X,Y] = joint_prob([data1,data2]);
% rows of X give values of X
x=X(1,:);
dx = x(2) - x(1);
% cols of Y give values of Y
y=Y(:,1);
dy = y(2) - y(1);

joint_density = joint_density/sum(sum(joint_density*dx*dy));
Hxy = -sum(sum(joint_density*dx*dy.*log(joint_density*dx*dy)));

info = Hx + Hy - Hxy;
 
end