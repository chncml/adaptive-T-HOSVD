function test_fixed_precision_with_low_rank_matrix_fix_vary
clc;clear;
n = 800;m = 800;d = [1 : n];
% d = d.^2;d = 1./d;
% d = exp(-d/7);
d = 0.001+(1+exp(d - 29)).^(-1);
u = randn(n);[u, ~] = qr(u, 0);
v = randn(m * m, n);[v, ~] = qr(v, 0);
A = u * diag(d) * v';normA = norm(A(:));
Aten = reshape(A, [n, m, m]);
tol = [1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 7.5e-3, 5e-3, 2.5e-3, 1e-3];
for k = 1 : length(tol)
    k
    err = (tol(k)^2) * (normA^2);
    for sample = 1 : 10
        sample;
        t1 = tic;[Q1, B1, k1] = randQB_EI_auto_bernoulli(Aten, err, 50, 1);t1 = toc(t1);
        t2 = tic;[Q2, B2, k2] = randQB_EI_auto_bernoulli_fix(Aten, err, 50, 1);t2 = toc(t2);
        t3 = tic;[Q3, B3, k3] = randQB_EI_auto_gaussian(Aten, err, 50, 1);t3 = toc(t3);
        t4 = tic;[Q4, B4, k4] = randQB_EI_auto_gaussian_fix(Aten, err, 50, 1);t4 = toc(t4);
        t5 = tic;[Q5, B5, k5] = randQB_EI_auto_kr_bernoulli(Aten, err, 50, 1);t5 = toc(t5);
        t6 = tic;[Q6, B6, k6] = randQB_EI_auto_kr_bernoulli_fix(Aten, err, 50, 1);t6 = toc(t6);
        t7 = tic;[Q7, B7, k7] = randQB_EI_auto_kr_gaussian(Aten, err, 50, 1);t7 = toc(t7);
        t8 = tic;[Q8, B8, k8] = randQB_EI_auto_kr_gaussian_fix(Aten, err, 50, 1);t8 = toc(t8);
        A1 = Q1 * B1;err1 = norm(A1(:) - A(:))/normA;
        A2 = Q2 * B2;err2 = norm(A2(:) - A(:))/normA;
        A3 = Q3 * B3;err3 = norm(A3(:) - A(:))/normA;
        A4 = Q4 * B4;err4 = norm(A4(:) - A(:))/normA;
        A5 = Q5 * B5;err5 = norm(A5(:) - A(:))/normA;
        A6 = Q6 * B6;err6 = norm(A6(:) - A(:))/normA;
        A7 = Q7 * B7;err7 = norm(A7(:) - A(:))/normA;
        A8 = Q8 * B8;err8 = norm(A8(:) - A(:))/normA;

        ERR1(k, sample) = err1;ERR2(k, sample) = err2;
        ERR3(k, sample) = err3;ERR4(k, sample) = err4;
        ERR5(k, sample) = err5;ERR6(k, sample) = err6;
        ERR7(k, sample) = err7;ERR8(k, sample) = err8;

        T1(k, sample) = t1;T2(k, sample) = t2;
        T3(k, sample) = t3;T4(k, sample) = t4;
        T5(k, sample) = t5;T6(k, sample) = t6;
        T7(k, sample) = t7;T8(k, sample) = t8;

        K1(k, sample) = k1;K2(k, sample) = k2;
        K3(k, sample) = k3;K4(k, sample) = k4;
        K5(k, sample) = k5;K6(k, sample) = k6;
        K7(k, sample) = k7;K8(k, sample) = k8;
    end
end
format short
[sum(T1, 2), sum(T2, 2), sum(T3, 2), sum(T4, 2), sum(T5, 2), sum(T6, 2), sum(T7, 2), sum(T8, 2)]/10
[sum(K1, 2), sum(K2, 2), sum(K3, 2), sum(K4, 2), sum(K5, 2), sum(K6, 2), sum(K7, 2), sum(K8, 2)]/10
format short e
[sum(ERR1, 2), sum(ERR2, 2), sum(ERR3, 2), sum(ERR4, 2), sum(ERR5, 2), sum(ERR6, 2), sum(ERR7, 2), sum(ERR8, 2)]/10
end