function test_fixed_precision_with_sthosvd_different_q
%% comparison on tensors from synthetic datasets
clear;clc;
n = 800;
b1 = randn(n);[b1, ~] = qr(b1);
b2 = randn(n);[b2, ~] = qr(b2);
b3 = randn(n);[b3, ~] = qr(b3);
v = [1 : n];tol = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4];

%% slow decay
v = v.^2;v = 1./v;

%% fast decay
% v = exp(-v/7);

%% S-shape decay
% v = 0.001+(1+exp(v - 29)).^(-1);

A = tendiag(v, [n, n, n]);
A = ttm(A, {b1, b2, b3}, [1, 2, 3]);
A = double(A);normA = norm(A(:));

ML11 = zeros(7, 3);ML12 = zeros(7, 3);ML13 = zeros(7, 3);ML14 = zeros(7, 3);ML15 = zeros(7, 3);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 0);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 2);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 3);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 4);t15 = toc(t15);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2)]/10

ML11 = zeros(7, 3);ML12 = zeros(7, 3);ML13 = zeros(7, 3);ML14 = zeros(7, 3);ML15 = zeros(7, 3);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 0);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 2);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 3);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 4);t15 = toc(t15);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2)]/10

ML11 = zeros(7, 3);ML12 = zeros(7, 3);ML13 = zeros(7, 3);ML14 = zeros(7, 3);ML15 = zeros(7, 3);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 0);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 2);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 3);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 4);t15 = toc(t15);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2)]/10

ML11 = zeros(7, 3);ML12 = zeros(7, 3);ML13 = zeros(7, 3);ML14 = zeros(7, 3);ML15 = zeros(7, 3);
for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 0);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 2);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 3);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 4);t15 = toc(t15);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2)]/10
end