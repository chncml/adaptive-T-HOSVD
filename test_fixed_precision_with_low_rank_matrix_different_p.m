function test_fixed_precision_with_low_rank_matrix_different_p
clc;clear;

n = 800;m = 800;d = [1 : n];
% d = d.^2;d = 1./d;
% d = exp(-d/7);
d = 0.001+(1+exp(d - 29)).^(-1);
u = randn(n);[u, ~] = qr(u, 0);
v = randn(m * m, n);[v, ~] = qr(v, 0);
A = u * diag(d) * v';normA = norm(A(:));
Aten = reshape(A, [n, m, m]);P = 1;
tol = [1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 7.5e-3, 5e-3, 2.5e-3, 1e-3];

T1 = [];T2 = [];T3 = [];T4 = [];T5 = [];
K1 = [];K2 = [];K3 = [];K4 = [];K5 = [];
ERR1 = [];ERR2 = [];ERR3 = [];ERR4 = [];ERR5 = [];

for k = 1 : length(tol)
    k;
    err = (tol(k)^2) * (normA^2);
    for sample = 1 : 10
        sample;
        t1 = tic;[Q1, B1, k1] = randQB_EI_auto_gaussian(Aten, err, 50, 0);t1 = toc(t1);
        t2 = tic;[Q2, B2, k2] = randQB_EI_auto_gaussian(Aten, err, 50, 1);t2 = toc(t2);
        t3 = tic;[Q3, B3, k3] = randQB_EI_auto_gaussian(Aten, err, 50, 2);t3 = toc(t3);
        t4 = tic;[Q4, B4, k4] = randQB_EI_auto_gaussian(Aten, err, 50, 3);t4 = toc(t4);
        t5 = tic;[Q5, B5, k5] = randQB_EI_auto_gaussian(Aten, err, 50, 4);t5 = toc(t5);

        A1 = Q1 * B1;err1 = norm(A1(:) - A(:))/normA;
        A2 = Q2 * B2;err2 = norm(A2(:) - A(:))/normA;
        A3 = Q3 * B3;err3 = norm(A3(:) - A(:))/normA;
        A4 = Q4 * B4;err4 = norm(A4(:) - A(:))/normA;
        A5 = Q5 * B5;err5 = norm(A5(:) - A(:))/normA;

        ERR1(k, sample) = err1;ERR2(k, sample) = err2;
        ERR3(k, sample) = err3;ERR4(k, sample) = err4;
        ERR5(k, sample) = err5;

        T1(k, sample) = t1;T2(k, sample) = t2;
        T3(k, sample) = t3;T4(k, sample) = t4;
        T5(k, sample) = t5;

        K1(k, sample) = k1;K2(k, sample) = k2;
        K3(k, sample) = k3;K4(k, sample) = k4;
        K5(k, sample) = k5;
    end
end
format short
[sum(T1, 2), sum(T2, 2), sum(T3, 2), sum(T4, 2), sum(T5, 2)]/10
[sum(K1, 2), sum(K2, 2), sum(K3, 2), sum(K4, 2), sum(K5, 2)]/10
format short e
[sum(ERR1, 2), sum(ERR2, 2), sum(ERR3, 2), sum(ERR4, 2), sum(ERR5, 2)]/10

T1 = [];T2 = [];T3 = [];T4 = [];T5 = [];
K1 = [];K2 = [];K3 = [];K4 = [];K5 = [];
ERR1 = [];ERR2 = [];ERR3 = [];ERR4 = [];ERR5 = [];
for k = 1 : length(tol)
    k;
    err = (tol(k)^2) * (normA^2);
    for sample = 1 : 10
        sample;
        t1 = tic;[Q1, B1, k1] = randQB_EI_auto_bernoulli(Aten, err, 50, 0);t1 = toc(t1);
        t2 = tic;[Q2, B2, k2] = randQB_EI_auto_bernoulli(Aten, err, 50, 1);t2 = toc(t2);
        t3 = tic;[Q3, B3, k3] = randQB_EI_auto_bernoulli(Aten, err, 50, 2);t3 = toc(t3);
        t4 = tic;[Q4, B4, k4] = randQB_EI_auto_bernoulli(Aten, err, 50, 3);t4 = toc(t4);
        t5 = tic;[Q5, B5, k5] = randQB_EI_auto_bernoulli(Aten, err, 50, 4);t5 = toc(t5);

        A1 = Q1 * B1;err1 = norm(A1(:) - A(:))/normA;
        A2 = Q2 * B2;err2 = norm(A2(:) - A(:))/normA;
        A3 = Q3 * B3;err3 = norm(A3(:) - A(:))/normA;
        A4 = Q4 * B4;err4 = norm(A4(:) - A(:))/normA;
        A5 = Q5 * B5;err5 = norm(A5(:) - A(:))/normA;

        ERR1(k, sample) = err1;ERR2(k, sample) = err2;
        ERR3(k, sample) = err3;ERR4(k, sample) = err4;
        ERR5(k, sample) = err5;

        T1(k, sample) = t1;T2(k, sample) = t2;
        T3(k, sample) = t3;T4(k, sample) = t4;
        T5(k, sample) = t5;

        K1(k, sample) = k1;K2(k, sample) = k2;
        K3(k, sample) = k3;K4(k, sample) = k4;
        K5(k, sample) = k5;
    end
end
format short
[sum(T1, 2), sum(T2, 2), sum(T3, 2), sum(T4, 2), sum(T5, 2)]/10
[sum(K1, 2), sum(K2, 2), sum(K3, 2), sum(K4, 2), sum(K5, 2)]/10
format short e
[sum(ERR1, 2), sum(ERR2, 2), sum(ERR3, 2), sum(ERR4, 2), sum(ERR5, 2)]/10

T1 = [];T2 = [];T3 = [];T4 = [];T5 = [];
K1 = [];K2 = [];K3 = [];K4 = [];K5 = [];
ERR1 = [];ERR2 = [];ERR3 = [];ERR4 = [];ERR5 = [];
for k = 1 : length(tol)
    k;
    err = (tol(k)^2) * (normA^2);
    for sample = 1 : 10
        sample;
        t1 = tic;[Q1, B1, k1] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 0);t1 = toc(t1);
        t2 = tic;[Q2, B2, k2] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 1);t2 = toc(t2);
        t3 = tic;[Q3, B3, k3] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 2);t3 = toc(t3);
        t4 = tic;[Q4, B4, k4] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 3);t4 = toc(t4);
        t5 = tic;[Q5, B5, k5] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 4);t5 = toc(t5);

        A1 = Q1 * B1;err1 = norm(A1(:) - A(:))/normA;
        A2 = Q2 * B2;err2 = norm(A2(:) - A(:))/normA;
        A3 = Q3 * B3;err3 = norm(A3(:) - A(:))/normA;
        A4 = Q4 * B4;err4 = norm(A4(:) - A(:))/normA;
        A5 = Q5 * B5;err5 = norm(A5(:) - A(:))/normA;

        ERR1(k, sample) = err1;ERR2(k, sample) = err2;
        ERR3(k, sample) = err3;ERR4(k, sample) = err4;
        ERR5(k, sample) = err5;

        T1(k, sample) = t1;T2(k, sample) = t2;
        T3(k, sample) = t3;T4(k, sample) = t4;
        T5(k, sample) = t5;

        K1(k, sample) = k1;K2(k, sample) = k2;
        K3(k, sample) = k3;K4(k, sample) = k4;
        K5(k, sample) = k5;
    end
end
format short
[sum(T1, 2), sum(T2, 2), sum(T3, 2), sum(T4, 2), sum(T5, 2)]/10
[sum(K1, 2), sum(K2, 2), sum(K3, 2), sum(K4, 2), sum(K5, 2)]/10
format short e
[sum(ERR1, 2), sum(ERR2, 2), sum(ERR3, 2), sum(ERR4, 2), sum(ERR5, 2)]/10

T1 = [];T2 = [];T3 = [];T4 = [];T5 = [];
K1 = [];K2 = [];K3 = [];K4 = [];K5 = [];
ERR1 = [];ERR2 = [];ERR3 = [];ERR4 = [];ERR5 = [];
for k = 1 : length(tol)
    k;
    err = (tol(k)^2) * (normA^2);
    for sample = 1 : 10
        sample;
        t1 = tic;[Q1, B1, k1] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 0);t1 = toc(t1);
        t2 = tic;[Q2, B2, k2] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 1);t2 = toc(t2);
        t3 = tic;[Q3, B3, k3] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 2);t3 = toc(t3);
        t4 = tic;[Q4, B4, k4] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 3);t4 = toc(t4);
        t5 = tic;[Q5, B5, k5] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 4);t5 = toc(t5);

        A1 = Q1 * B1;err1 = norm(A1(:) - A(:))/normA;
        A2 = Q2 * B2;err2 = norm(A2(:) - A(:))/normA;
        A3 = Q3 * B3;err3 = norm(A3(:) - A(:))/normA;
        A4 = Q4 * B4;err4 = norm(A4(:) - A(:))/normA;
        A5 = Q5 * B5;err5 = norm(A5(:) - A(:))/normA;

        ERR1(k, sample) = err1;ERR2(k, sample) = err2;
        ERR3(k, sample) = err3;ERR4(k, sample) = err4;
        ERR5(k, sample) = err5;

        T1(k, sample) = t1;T2(k, sample) = t2;
        T3(k, sample) = t3;T4(k, sample) = t4;
        T5(k, sample) = t5;

        K1(k, sample) = k1;K2(k, sample) = k2;
        K3(k, sample) = k3;K4(k, sample) = k4;
        K5(k, sample) = k5;
    end
end
format short
[sum(T1, 2), sum(T2, 2), sum(T3, 2), sum(T4, 2), sum(T5, 2)]/10
[sum(K1, 2), sum(K2, 2), sum(K3, 2), sum(K4, 2), sum(K5, 2)]/10
format short e
[sum(ERR1, 2), sum(ERR2, 2), sum(ERR3, 2), sum(ERR4, 2), sum(ERR5, 2)]/10
end