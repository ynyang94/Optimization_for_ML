function y=Amijo(w,c,tk,alpha,b,A,A0)
%% Amijo condition for gradient.
n = 15;
x=[w;c];
grad=grad_logi(w,c,b,A);
x_new=x-tk*grad;
w1=x_new(1:n);
c1=x_new(n+1);
%%
func_value=logistic(w1,c1,b,A0);
quan1=logistic(w,c,b,A0)-alpha*tk*norm(grad,2)^2;
%%
y=(func_value <= quan1);
end