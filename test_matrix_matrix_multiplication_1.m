function test_matrix_matrix_multiplication_1
clc;clear;
dim = 800;r1 = [20:20:200];
t11 = [];t12 = [];t13 = [];t14 = [];
Amat = randn(dim, dim^2);
for k = 1 : length(r1)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim, r1(k));y = randn(dim, r1(k));
        z1 = kr(x, y);B1 = Amat * z1;
        t = toc(t);t11(k, sample) = t;
        t = tic;z2 = randn(dim^2, r1(k));
        B2 = Amat * z2;
        t = toc(t);t12(k, sample) = t;
        t = tic;
        x = randi(2, [dim, r1(k)]) * 2 - 3;
        y = randi(2, [dim, r1(k)]) * 2 - 3;
        z3 = kr(x, y);B3 = Amat * z3;
        t = toc(t);t13(k, sample) = t;
        t = tic;z4 = randi(2, [dim^2, r1(k)]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t14(k, sample) = t;
    end
end
dim1 = 200;r2 = [10 : 10 : 100];
Amat = randn(dim1, dim1^3);
t21 = [];t22 = [];t23 = [];t24 = [];
for k = 1 : length(r2)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1, r2(k));
        x2 = randn(dim1, r2(k));
        x3 = randn(dim1, r2(k));
        z1 = kr(x1, x2);z1 = kr(z1, x3);
        B1 = Amat * z1;
        t = toc(t);t21(k, sample) = t;
        t = tic;z2 = randn(dim1^3, r2(k));
        B2 = Amat * z2;
        t = toc(t);t22(k, sample) = t;
        t = tic;
        x1 = randi(2, [dim1, r2(k)]) * 2 - 3;
        x2 = randi(2, [dim1, r2(k)]) * 2 - 3;
        x3 = randi(2, [dim1, r2(k)]) * 2 - 3;
        z3 = kr(x1, x2);z3 = kr(z3, x3);
        B3 = Amat * z3;
        t = toc(t);t23(k, sample) = t;
        t = tic;z4 = randi(2, [dim1^3, r2(k)]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t24(k, sample) = t;
    end
end
dim = 800;r1 = [20:20:200];
A1 = randn(dim, dim);A2 = randn(dim, dim);
Amat = kr(A1, A2);Amat = Amat';
t31 = [];t32 = [];t33 = [];t34 = [];
for k = 1 : length(r1)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim, r1(k));
        y = randn(dim, r1(k));
        B1 = (A1'*x).*(A2'*y);
        t = toc(t);t31(k, sample) = t;
        t = tic;z2 = randn(dim^2, r1(k));
        B2 = Amat * z2;
        t = toc(t);t32(k, sample) = t;
        t = tic;
        x = randi(2, [dim, r1(k)]) * 2 - 3;
        y = randi(2, [dim, r1(k)]) * 2 - 3;
        B3 = (A1'*x).*(A2'*y);
        t = toc(t);t33(k, sample) = t;
        t = tic;z4 = randi(2, [dim^2, r1(k)]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t34(k, sample) = t;
    end
end
dim1 = 200;r2 = [10:10:100];
A1 = randn(dim1);A2 = randn(dim1);A3 = randn(dim1);
Amat = kr(A1, kr(A2, A3));Amat = Amat';
t41 = [];t42 = [];t43 = [];t44 = [];
for k = 1 : length(r2)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1, r2(k));x2 = randn(dim1, r2(k));
        x3 = randn(dim1, r2(k));
        B1 = (A1' * x1).*(A2' * x2).*(A3' * x3);
        t = toc(t);t41(k, sample) = t;
        t = tic;z2 = randn(dim1^3, r2(k));
        B2 = Amat * z2;
        t = toc(t);t42(k, sample) = t;
        t = tic;
        x1 = randi(2, [dim1, r2(k)]) * 2 - 3;
        x2 = randi(2, [dim1, r2(k)]) * 2 - 3;
        x3 = randi(2, [dim1, r2(k)]) * 2 - 3;
        B3 = (A1' * x1).*(A2' * x2).*(A3' * x3);
        t = toc(t);t43(k, sample) = t;
        t = tic;z4 = randi(2, [dim1^3, r2(k)]) * 2 - 3;
        B4 = Amat * z4;
        t = toc(t);t44(k, sample) = t;
    end
end
format short e
[sum(t11, 2)/10, sum(t12, 2)/10, sum(t13, 2)/10, sum(t14, 2)/10]
[sum(t21, 2)/10, sum(t22, 2)/10, sum(t23, 2)/10, sum(t24, 2)/10]
[sum(t31, 2)/10, sum(t32, 2)/10, sum(t33, 2)/10, sum(t34, 2)/10]
[sum(t41, 2)/10, sum(t42, 2)/10, sum(t43, 2)/10, sum(t44, 2)/10]

figure(1)

subplot(1, 2, 1)
semilogy(r1, sum(t11, 2)/10, '-.*');hold on
semilogy(r1, sum(t12, 2)/10, '-.d');hold on
semilogy(r1, sum(t13, 2)/10, '-.+');hold on
semilogy(r1, sum(t14, 2)/10, '-.o');hold on
subplot(1, 2, 2)
semilogy(r2, sum(t21, 2)/10, '-.*');hold on
semilogy(r2, sum(t22, 2)/10, '-.d');hold on
semilogy(r2, sum(t23, 2)/10, '-.+');hold on
semilogy(r2, sum(t24, 2)/10, '-.o');hold on

figure(2)

subplot(1, 2, 1)
semilogy(r1, sum(t31, 2)/10, '-.*');hold on
semilogy(r1, sum(t32, 2)/10, '-.d');hold on
semilogy(r1, sum(t33, 2)/10, '-.+');hold on
semilogy(r1, sum(t34, 2)/10, '-.o');hold on
subplot(1, 2, 2)
semilogy(r2, sum(t41, 2)/10, '-.*');hold on
semilogy(r2, sum(t42, 2)/10, '-.d');hold on
semilogy(r2, sum(t43, 2)/10, '-.+');hold on
semilogy(r2, sum(t44, 2)/10, '-.o');hold on
end