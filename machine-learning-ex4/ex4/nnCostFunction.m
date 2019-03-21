function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% add ones to the X data matrix
X = [ones(m, 1) X]; %dimension 5000 * 401

% calculate a2 hidden layer matrix
z2 = X * Theta1';   % dimension 5000 * 25
a2 = sigmoid(z2);   % dimension 5000 * 25

% add ones to the a2 hidden layer matrix
a2 = [ones(size(a2, 1), 1) a2]; % dimension 5000 * 26

% calculate output layer h_x
z3 = a2*Theta2'; % dimension 5000 * 10
h_x = sigmoid(z3); % dimension 5000 * 10

% calculate the cost function

z = 1:num_labels;  %create a row vector for sequence of labels (dimension 1 * 10)
Y = y == z;        %create a matrix for output layer (dimension 5000 * 10)
J = (1/m)*sum(sum(- Y .* log(h_x) - (1 - Y) .* log(1 - h_x)));


%J = (1/m) * sum( sum(
%		     (-Y.*log(h_x))-((1-Y).*log(1-h_x))
%		   )
%	       );  %scalar


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


for t = 1:m
  %Step 1
  % calculate a2 hidden layer vector
  z2 = X(t, :)*Theta1'; % dimension 1*401 times 401*25 = 1 * 25
  a2 = sigmoid(z2);    % dimension 1 * 25
  % add ones to the a2 hidden layer vector
  a2 = [1; a2'];       % dimension 26 * 1
  % calculate output layer h_x
  z3 = Theta2*a2; % dimension 10*1
  h_x = sigmoid(z3); % dimension  10*1
  %Step 2
  % calculate the difference of the output layer and the true value
  z = 1:num_labels;  %create a row vector for sequence of labels (dimension 1 * 10)
  Y = y == z;        %create a matrix for output layer (dimension 5000 * 10)
  delta3 =  h_x - Y(t, :)';  %difference between activation and the true value (dimension 10*1)
  %Step 3
  % the difference in the hidden layer
  delta2 = Theta2(:, 2:end)'*delta3 .* sigmoidGradient(z2)';   % dimension 25 * 1
  %Step 4
  % compute the gradients for theta1 and theta2 separately
  Theta1_grad = Theta1_grad + delta2*(X(t, :)); % dimension 25 * 401
  Theta2_grad = Theta2_grad + delta3*(a2'); % dimension 10 * 26
endfor

Theta1_grad = (1/m)*Theta1_grad; % dimension 25*401
Theta2_grad = (1/m)*Theta2_grad; % dimension 10*26

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% regularization term
reg_term = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^2)));
  

J = J + reg_term;


% regularization term for gradients 
reg_termTheta1 = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]; % 25*401
reg_termTheta2 = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]; % 10*26


% adding regularization term to Theta_grad

Theta1_grad = Theta1_grad + reg_termTheta1;
Theta2_grad = Theta2_grad + reg_termTheta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
