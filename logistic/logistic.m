function y=logistic(w,c,b,A0)
%% Parameter Explaination
%w: weight paramter vector: R^n
%c: Bias term
%b: Output label, {+1,-1} \in R^m
%A0: Original input \in R^{m by n}

%%
m=length(b);
count=0;
for i=1:m
    count=count+log(1+exp(-b(i,1)*(dot(A0(i,:),w)+c)));
end
y=1/m*count;
end