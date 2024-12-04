A = [[0 1 0 0];[0 -0.1818 2.6730 0];[0 0 0 1];[0 -0.4545 -31.1800 0]];
B = [0;1.818;0;4.545];
C = [1 0 0 0];
delta  = 0.001;
tspan = 0:delta:6-delta;
u = sin(tspan);
test_u = sin(tspan(2001:6000));
% Define initial condition for x 
x0 = [0; 0; 0; 0];

% Define the system of ODEs
odefun = @(tspan, x) A * x + B * sin(tspan);

% Solve using ode45
[t, x] = ode45(odefun, tspan, x0);
x = x';
y = C*x;
y_train  = y(1:2000);
y_test = y(2001:end);
r = 4;
q = 4;
p = 2000 - q;
% Predefine Hankel matrix dimensions
H = zeros(q, p);
for i = 1:p
    for j = 1:q
        H(j, i) = y_train(i + j - 1);
    end
end
%SVD
[U,S,V] = svd(H);
Ur = U(:,1:r);
Vr = V(:,1:r);
Sr = S(1:r,1:r);
Vr_T = Vr';
Hr = Ur*Sr*Vr';
disp(Ur*Sr);
error = norm(H - Hr,'fro');
% Compute time derivatives of Vr
Vr_dot = zeros(size(Vr));
for i = 2:(size(Vr, 1)-1)
    Vr_dot(i, :) = (Vr(i+1, :) - Vr(i-1, :)) / (2 * delta);  % Central difference
end

% Forward and backward difference for boundaries
Vr_dot(1, :) = (Vr(2, :) - Vr(1, :)) / delta;
Vr_dot(end, :) = (Vr(end, :) - Vr(end-1, :)) / delta;
ur = sin(tspan(1:p)); 
% Construct Theta(Vr)
Theta_Vr = [Vr, ur']; 

numObservations = 1996;
numPredictors = 5;
lambda = 0.001;
%%
% L1-regularized LAD regression using CVX (vectorized, with explicit L1 norm)
cvx_begin
    variable xi(4, 5);
    minimize(norm(Vr_dot - Theta_Vr*xi',1))
cvx_end

% Display the results
disp('Estimated coefficients (CVX):');
disp(xi);
%%
A_hat = xi(1:4,1:4);
B_hat = xi(1:4,5);
C_hat = Ur*Sr;
C_hat = C_hat(1,:);
%%
% Initialize state vector for the identified model
x_identified = zeros(size(C_hat, 2), length(tspan));  % State vector
x_identified(:, 1) = x0;  % Initial condition

% Simulate dynamics
for i = 2:length(tspan)
    x_identified(:, i) = A_hat * x_identified(:, i-1) + B_hat * u(i-1);
end

% Compute the output
y_identified = C_hat * x_identified;

% Compare with true data
figure;
plot(tspan, y, 'b', 'DisplayName', 'True Output');
hold on;
plot(tspan, y_identified, 'r--', 'DisplayName', 'Identified Output');
legend();
title('True vs Identified Output');
xlabel('Time');
ylabel('Output');
%%
% Compute eigenvalues of A and A_hat
eig_A = eig(A);  % True system matrix
eig_A_hat = eig(A_hat);  % Identified system matrix

% Plot eigenvalues
figure;
scatter(real(eig_A), imag(eig_A), 'b', 'DisplayName', 'True Eigenvalues');
hold on;
scatter(real(eig_A_hat), imag(eig_A_hat), 'r', 'DisplayName', 'Identified Eigenvalues');
xlabel('Real Part');
ylabel('Imaginary Part');
title('Eigenvalue Comparison');
legend();
grid on;
%%
test_time = tspan(2001:6000);
% Initialize state vector for test data
x_test = zeros(size(C_hat, 2), length(test_u));
x_test(:, 1) = x_identified(:, end);  % Start from the last state of training

% Simulate test dynamics
for i = 2:length(test_u)
    x_test(:, i) = A_hat * x_test(:, i-1) + B_hat * test_u(i-1);
end

% Compute the output
y_test_identified = C_hat * x_test;

% Compare with true test data
figure;
plot(test_time, y_test, 'b', 'DisplayName', 'True Test Output');
hold on;
plot(test_time, y_test_identified, 'r--', 'DisplayName', 'Identified Test Output');
legend();
title('Test Data: True vs Identified Output');
xlabel('Time');
ylabel('Output');
%%
% Compute error
error_train = y - y_identified;
error_test = y_test - y_test_identified;

% Plot error
figure;
subplot(2, 1, 1);
plot(time, error_train, 'k');
title('Training Error');
xlabel('Time');
ylabel('Error');

subplot(2, 1, 2);
plot(test_time, error_test, 'k');
title('Test Error');
xlabel('Time');
ylabel('Error');
%%
% Compute energy contributions
singular_values = diag(Sigma);
energy = cumsum(singular_values.^2) / sum(singular_values.^2);

% Plot cumulative energy
figure;
plot(energy, 'o-');
title('Cumulative Energy Captured by Modes');
xlabel('Number of Modes');
ylabel('Cumulative Energy');











