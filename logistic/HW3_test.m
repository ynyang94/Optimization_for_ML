clear all
%
rng(1234)
T = readtable('framingham_clean.csv');
train_data = T(1:2656,:);
test_data = T(2657:3656,:);
A0 = table2array(train_data(:,1:15));
A0_test = table2array(test_data(:,1:15));
b_test = table2array(test_data(:,16)); 
b = table2array(train_data(:,16));
b(b==0)=-1;
b_test(b_test == 0) = -1;
n = 15;
m=2656;
A = zeros(2656,15);
for i = 1:15
    A(:,i) = A0(:,i).*b;
end

%% Some useful hyperparameter for gradient descent and BFGS.

alpha=0.1;
beta=0.8;
%epsilon can be 1e-8; 1e-10;1e-12
epsilon=1e-2;
%% Initialization of parameter for gradient descent.
w0=randn(n,1);
c0=0;
grad1=grad_logi(w0,c0,b,A);
stop1=norm(grad1,2)^2;
fun_val = logistic(w0, c0, b, A0);
tk=1;
correct = 0;
test_output = test_label(w0,c0,A0_test);
for i=1:1000
    if test_output(i) == b_test(i)
        correct = correct+1;
    end
end
accuracy = correct/1000;
%% Loop starts!!
count=0;
%mem1=[];
mem2 = [];
%mem1=[mem1;stop1];
mem2 = [mem2, fun_val];
mem3 = [];
mem3 = [mem3,accuracy];
max_iter = 2000;
while stop1>epsilon && count < max_iter
    %Check Amijo condition
    log_value=Amijo(w0,c0,tk,alpha,b,A,A0);
    while log_value~=1
        tk=beta*tk;
        log_value=Amijo(w0,c0,tk,alpha,b,A,A0);
    end
    %Gradient update for w,c
    w0=w0-tk*grad1(1:n);
    c0=c0-tk*grad1(n+1);
    %Compute new gradient.
    grad1=grad_logi(w0,c0,b,A);
    % Compute new norm of gradient.
    stop1=norm(grad1,2)^2;
    fun_val = logistic(w0,c0,b,A0);
    %collect norm of gradient for figure plotting.
    correct = 0;
    test_output = test_label(w0,c0,A0_test);
    for i=1:1000
        if test_output(i) == b_test(i)
            correct = correct+1;
        end
    end
    accuracy = correct/1000;
    %mem1=[mem1;stop1];
    mem2 = [mem2; fun_val];
    mem3 = [mem3; accuracy];
    %Add count.
    count=count+1;
end

correct = 0;
test_output = test_label(w0,c0,A0_test);
for i=1:1000
    if test_output(i) == b_test(i)
        correct = correct+1;
    end
end
accuracy = correct/1000;


%% Plot
% figure(1)
% loglog(mem1,'-*','LineWidth',1.5)
% xlabel('number of iterations')
% ylabel('norm of gradient')
% title('loglog plot for squared norm of gradient with \epsilon=1e-2')
figure(2)
loglog(mem2, '-*', 'LineWidth', 1.5)
xlabel('number of iterations')
ylabel('loss function')
title('loglog plot for loss function plot')
figure(3)
plot(mem3, '-*')
xlabel('iterations')
ylabel('accuracy')
title('test error curve')
