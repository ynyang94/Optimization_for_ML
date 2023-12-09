function y=Amijo2(w,c,tk,alpha,pk,b,A,A0)
%% Amijo condition for BFGS.
x=[w;c];
grad=grad_logi(w,c,b,A);
x_new=x+tk*pk;
w1=x_new(1:10);
c1=x_new(11);
%%
func_value=logistic(w1,c1,b,A0);
quan1=logistic(w,c,b,A0)+alpha*tk*grad'*pk;
%%
y=(func_value <= quan1);
end