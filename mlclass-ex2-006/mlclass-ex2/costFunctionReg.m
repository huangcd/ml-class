function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
theta_eye = theta;
theta_eye(1) = 0;
g = sigmoid(X*theta);
J = -1/m*(sum(y.*log(g) .+ (1.-y).*log(1.-g))) + lambda / 2 / m * sum(theta_eye .^ 2);
grad = (1/m).*(X'*(g .- y)) .+ (lambda / m) .* theta_eye;

end
