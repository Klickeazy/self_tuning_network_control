% scratch code experiments for self-tuning network control architectures

% tyler summers, feb 4, 2022



%% problem data

n = 50; % number of nodes/states

k = 2; % number of actuators per time step



A = randn(n); A = 1.2*A/max(abs(eig(A))); % dynamics matrix

Bset = eye(n);  % set of input vectors



% load SelfTuningEx2.mat



Q = eye(n); % state cost

R = eye(k); % input cost (uniform for all actuators)



% arbitrary set of actuators

Bfixed = Bset(:,1:k);

[Pfixed, Kfixed] = idare(A,Bfixed,Q,R);



%% evaluate performance of fixed architecture

x0 = 5*randn(n,1);

Jfixed = x0'*Pfixed*x0;

T = 50;

Jsfixed = zeros(T+1,1);



W = 0.01*randn(n,T);

x = zeros(n, T+1);

x(:,1) = x0;

for t=1:T

    u = -Kfixed*x(:,t);

    Jsfixed(t) = x(:,t)'*Q*x(:,t) + u'*R*u;

    x(:,t+1) = A*x(:,t) + Bfixed*u + W(:,t);

end

Jsfixed(T+1) = x(:,T+1)'*Q*x(:,T+1);

Jfixed = sum(Jsfixed);





%% plotting

figure;

cmap = [186 193 184;

         88 164 176;

         12 124  89;

         43  48  58;

        214  73  51]/255;

subplot(2,1,1);

grid on;

set(gca, 'colororder', cmap, 'NextPlot', 'replacechildren', 'FontSize', 18, 'FontName', 'Times New Roman');

stairs(0:T,x', 'LineWidth', 2);

xlabel('time (s)');

ylabel('network states');



%% evaluate performance of self-tuning architecture

x = zeros(n, T+1);

x(:,1) = x0;

J = zeros(T+1,1);

for t = 1:T

    % greedily find top k actuators for current state

    [Bk, Kk] = greedy(A, Bset, Q, x(:,t), k);

    find(Bk == 1)

    u = -Kk*x(:,t);

    J(t) = x(:,t)'*Q*x(:,t) + u'*R*u;

    x(:,t+1) = A*x(:,t) + Bk*u + W(:,t);

%     t

end

J(T+1) = x(:,T+1)'*Q*x(:,T+1);



Jselftuning = sum(J);



%% plotting

cmap = [186 193 184;

         88 164 176;

         12 124  89;

         43  48  58;

        214  73  51]/255;



% figure;

subplot(2,1,2);

grid on;

set(gca, 'colororder', cmap, 'NextPlot', 'replacechildren', 'FontSize', 18, 'FontName', 'Times New Roman');

stairs(0:T,x', 'LineWidth', 2);

xlabel('time (s)');

ylabel('network states');



figure;

grid on;

stairs(Jsfixed, 'LineWidth', 2); hold on; stairs(J, 'LineWidth', 2);

PerformanceImprovement = (Jfixed - Jselftuning)/Jselftuning*100

xlabel('time (s)');

ylabel('cost');

legend('fixed architecture', 'self-tuning architecture');

set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');

linkaxes;





%% utility

function [Bk, Kk] = greedy(A, Bset, Q, x, k)

    % this optimizes LQR cost associated with a specific state

    n = size(A,1);

    S = 1:n;

    Sstar = [];

    for i=1:k

        J = zeros(length(S),1);

        for j=1:length(S)

            P = idare(A, Bset(:,[Sstar S(j)]), Q, eye(length(Sstar)+1));

            J(j) = x'*P*x;

        end

        [Jstar, istar] = min(J);
        Jstar

        Sstar = [Sstar S(istar)];

        S = setdiff(S, S(istar));

        i;

    end

    Bk = Bset(:,Sstar);

    [Pk, Kk] = idare(A, Bk, Q, eye(k));

end