function test_matrix_matrix_multiplication
dim = [50 : 50 : 800];
r = 10;
t11 = [];t12 = [];t13 = [];t14 = [];
for k = 1 : length(dim)
    k
    A = randn(dim(k), dim(k), dim(k));
    Amat = reshape(A, dim(k), []);
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim(k), r);y = randn(dim(k), r);
        z1 = kr(x, y);B1 = Amat * z1;
        t = toc(t);t11(k, sample) = t;
        t = tic;z2 = randn(dim(k)^2, r);
        B2 = Amat * z2;
        t = toc(t);t12(k, sample) = t;
        t = tic;
        x = randi(2, [dim(k), r]) * 2 - 3;
        y = randi(2, [dim(k), r]) * 2 - 3;
        z3 = kr(x, y);B3 = Amat * z3;
        t = toc(t);t13(k, sample) = t;
        t = tic;z4 = randi(2, [dim(k)^2, r]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t14(k, sample) = t;
    end
end
dim1 = [30 : 30 : 300];
t21 = [];t22 = [];t23 = [];t24 = [];
for k = 1 : length(dim1)
    k
    Amat = randn(dim1(k), dim1(k)^3);
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1(k), r);x2 = randn(dim1(k), r);x3 = randn(dim1(k), r);
        z1 = kr(x1, x2);z1 = kr(z1, x3);
        B1 = Amat * z1;
        t = toc(t);t21(k, sample) = t;
        t = tic;z2 = randn(dim1(k)^3, r);
        B2 = Amat * z2;
        t = toc(t);t22(k, sample) = t;
        t = tic;
        x1 = randi(2, [dim1(k), r]) * 2 - 3;
        x2 = randi(2, [dim1(k), r]) * 2 - 3;
        x3 = randi(2, [dim1(k), r]) * 2 - 3;
        z3 = kr(x1, x2);z3 = kr(z3, x3);
        B3 = Amat * z3;
        t = toc(t);t23(k, sample) = t;
        t = tic;z4 = randi(2, [dim1(k)^3, r]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t24(k, sample) = t;
    end
end
dim = [50 : 50 : 800];
r = 10;
t31 = [];t32 = [];t33 = [];t34 = [];
for k = 1 : length(dim)
    k
    A1 = randn(dim(k), dim(k));A2 = randn(dim(k), dim(k));
    Amat = kr(A1, A2);Amat = Amat';
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim(k), r);y = randn(dim(k), r);
        B1 = (A1'*x).*(A2'*y);
        t = toc(t);t31(k, sample) = t;
        t = tic;z2 = randn(dim(k)^2, r);
        B2 = Amat * z2;
        t = toc(t);t32(k, sample) = t;
        t = tic;
        x = randi(2, [dim(k), r]) * 2 - 3;
        y = randi(2, [dim(k), r]) * 2 - 3;
        B3 = (A1'*x).*(A2'*y);
        t = toc(t);t33(k, sample) = t;
        t = tic;z4 = randi(2, [dim(k)^2, r]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t34(k, sample) = t;
    end
end
dim1 = [30 : 30 : 300];
t41 = [];t42 = [];t43 = [];t44 = [];
for k = 1 : length(dim1)
    k
    A1 = randn(dim1(k));A2 = randn(dim1(k));A3 = randn(dim1(k));
    Amat = kr(A1, kr(A2, A3));Amat = Amat';
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1(k), r);x2 = randn(dim1(k), r);x3 = randn(dim1(k), r);
        B1 = (A1' * x1).*(A2' * x2).*(A3' * x3);
        t = toc(t);t41(k, sample) = t;
        t = tic;z2 = randn(dim1(k)^3, r);
        B2 = Amat * z2;
        t = toc(t);t42(k, sample) = t;
        t = tic;
        x1 = randi(2, [dim1(k), r]) * 2 - 3;
        x2 = randi(2, [dim1(k), r]) * 2 - 3;
        x3 = randi(2, [dim1(k), r]) * 2 - 3;
        B3 = (A1' * x1).*(A2' * x2).*(A3' * x3);
        t = toc(t);t43(k, sample) = t;
        t = tic;z4 = randi(2, [dim1(k)^3, r]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t44(k, sample) = t;
    end
end
format short e
[sum(t11, 2)/10, sum(t12, 2)/10, sum(t13, 2)/10, sum(t14, 2)/10]
[sum(t21, 2)/10, sum(t22, 2)/10, sum(t23, 2)/10, sum(t24, 2)/10]
[sum(t31, 2)/10, sum(t32, 2)/10, sum(t33, 2)/10, sum(t34, 2)/10]
[sum(t41, 2)/10, sum(t42, 2)/10, sum(t43, 2)/10, sum(t44, 2)/10]
end