function y=grad_logi(w,c,b,A)
%%
%w: weight parameter vector:R^n
%c: Bias term
%b: output label:R^m
%A= A0.*b, the element wise product of original data input w.r.t label set.
%%
m=length(b);
n=length(w);
e=ones(m,1);
p=e./(e+exp(-A*w-c.*b));
y1=-(1/m)*b'*(e-p);
y2=-(1/m)*A'*(e-p);
y=[y2;y1];
end