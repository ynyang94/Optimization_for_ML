function output = test_label(w,c,A0_test)
matrix_size = size(A0_test);
m = matrix_size(1);
n = matrix_size(2);
output = zeros(m,1);
for i=1:m
    P1 = 1/(1+exp(-(A0_test(i,:)*w+c)));
    P0 = 1/(1+exp(A0_test(i,:)*w+c));
    if P1 > P0
        output(i) = 1;
    else
        output(i) = -1;
    end
end

    

