function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h_theta = sigmoid(X * theta);

% The below code creates a new vector by replacing the first row in theta by zeros
% and retaining the rest of the rows as it is
% The reason for this formatting is we peanalize all the parameters with the
% regularized value except for the first parameter

theta_without_zero = [ [ 0 ]; theta(2:length(theta))];

cost_of_regularized_parameter = (lambda/(2*m))* sum(theta_without_zero.^2);

J = (1/m)* sum( -y .* log(h_theta) - (1-y) .* log( 1 - h_theta)) + cost_of_regularized_parameter;

grad = (1/m)* sum((h_theta - y) .* X) + (lambda / m) * theta_without_zero';


% =============================================================

end
