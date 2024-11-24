function test_different_processing_order
%% comparison on tensors from synthetic datasets
clear;clc;
n = 800;
b1 = randn(n);[b1, ~] = qr(b1);
b2 = randn(n);[b2, ~] = qr(b2);
b3 = randn(n);[b3, ~] = qr(b3);
v = [1 : n];tol = [1e-1, 1e-2, 1e-3];

% slow decay
v = v.^2;v = 1./v;


A = tendiag(v, [n, n, n]);
A = ttm(A, {b1, b2, b3}, [1, 2, 3]);
A = double(A);normA = norm(A(:));
A1 = permute(A, [1, 2, 3]);
A2 = permute(A, [1, 3, 2]);
A3 = permute(A, [2, 1, 3]);
A4 = permute(A, [2, 3, 1]);
A5 = permute(A, [3, 1, 2]);
A6 = permute(A, [3, 2, 1]);

%% ST-HOSVD (Gaussian)
ML11 = zeros(3, 3);ML12 = zeros(3, 3);ML13 = zeros(3, 3);ML14 = zeros(3, 3);ML15 = zeros(3, 3);ML16 = zeros(3, 3);
T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);T14 = zeros(3,1);T15 = zeros(3,1);T16 = zeros(3,1);
ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);ERR14 = zeros(3,1);ERR15 = zeros(3,1);ERR16 = zeros(3,1);
for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_gaussian(A1, relerr, 60, 1);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_gaussian(A2, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_gaussian(A3, relerr, 60, 1);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_gaussian(A4, relerr, 60, 1);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_gaussian(A5, relerr, 60, 1);t15 = toc(t15);
        t16 = tic;[G16, U16, mult_rank16] = adap_randomized_shosvd_EI_gaussian(A6, relerr, 60, 1);t16 = toc(t16);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A1(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A2(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A3(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A4(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A5(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        A16 = tmprod(G16, U16, [1, 2, 3]);Err16 = norm(A16(:) - A6(:))/normA;ERR16(k, sample) = Err16;T16(k, sample) = t16;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
        ML16(k, :) = ML16(k, :) + mult_rank16;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15, ML16]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2), sum(T16, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2), sum(ERR16, 2)]/10

%% ST-HOSVD (Uniform)
ML11 = zeros(3, 3);ML12 = zeros(3, 3);ML13 = zeros(3, 3);ML14 = zeros(3, 3);ML15 = zeros(3, 3);ML16 = zeros(3, 3);
T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);T14 = zeros(3,1);T15 = zeros(3,1);T16 = zeros(3,1);
ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);ERR14 = zeros(3,1);ERR15 = zeros(3,1);ERR16 = zeros(3,1);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_uniform(A1, relerr, 60, 1);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_uniform(A2, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_uniform(A3, relerr, 60, 1);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_uniform(A4, relerr, 60, 1);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_uniform(A5, relerr, 60, 1);t15 = toc(t15);
        t16 = tic;[G16, U16, mult_rank16] = adap_randomized_shosvd_EI_uniform(A6, relerr, 60, 1);t16 = toc(t16);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A1(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A2(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A3(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A4(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A5(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        A16 = tmprod(G16, U16, [1, 2, 3]);Err16 = norm(A16(:) - A6(:))/normA;ERR16(k, sample) = Err16;T16(k, sample) = t16;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
        ML16(k, :) = ML16(k, :) + mult_rank16;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15, ML16]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2), sum(T16, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2), sum(ERR16, 2)]/10

%% ST-HOSVD (KR-Gaussian)
ML11 = zeros(3, 3);ML12 = zeros(3, 3);ML13 = zeros(3, 3);ML14 = zeros(3, 3);ML15 = zeros(3, 3);ML16 = zeros(3, 3);
T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);T14 = zeros(3,1);T15 = zeros(3,1);T16 = zeros(3,1);
ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);ERR14 = zeros(3,1);ERR15 = zeros(3,1);ERR16 = zeros(3,1);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_kr_gaussian(A1, relerr, 60, 1);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_gaussian(A2, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_gaussian(A3, relerr, 60, 1);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_kr_gaussian(A4, relerr, 60, 1);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_kr_gaussian(A5, relerr, 60, 1);t15 = toc(t15);
        t16 = tic;[G16, U16, mult_rank16] = adap_randomized_shosvd_EI_kr_gaussian(A6, relerr, 60, 1);t16 = toc(t16);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A1(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A2(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A3(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A4(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A5(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        A16 = tmprod(G16, U16, [1, 2, 3]);Err16 = norm(A16(:) - A6(:))/normA;ERR16(k, sample) = Err16;T16(k, sample) = t16;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
        ML16(k, :) = ML16(k, :) + mult_rank16;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15, ML16]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2), sum(T16, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2), sum(ERR16, 2)]/10

%% ST-HOSVD (KR-Uniform)
ML11 = zeros(3, 3);ML12 = zeros(3, 3);ML13 = zeros(3, 3);ML14 = zeros(3, 3);ML15 = zeros(3, 3);ML16 = zeros(3, 3);
T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);T14 = zeros(3,1);T15 = zeros(3,1);T16 = zeros(3,1);
ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);ERR14 = zeros(3,1);ERR15 = zeros(3,1);ERR16 = zeros(3,1);

for k = 1 : length(tol)
    k;
    relerr = tol(k);
    for sample = 1 : 10
        sample;
        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_EI_kr_uniform(A1, relerr, 60, 1);t11 = toc(t11);
        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_uniform(A2, relerr, 60, 1);t12 = toc(t12);
        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_uniform(A3, relerr, 60, 1);t13 = toc(t13);
        t14 = tic;[G14, U14, mult_rank14] = adap_randomized_shosvd_EI_kr_uniform(A4, relerr, 60, 1);t14 = toc(t14);
        t15 = tic;[G15, U15, mult_rank15] = adap_randomized_shosvd_EI_kr_uniform(A5, relerr, 60, 1);t15 = toc(t15);
        t16 = tic;[G16, U16, mult_rank16] = adap_randomized_shosvd_EI_kr_uniform(A6, relerr, 60, 1);t16 = toc(t16);
        A11 = tmprod(G11, U11, [1, 2, 3]);Err11 = norm(A11(:) - A1(:))/normA;ERR11(k, sample) = Err11;T11(k, sample) = t11;
        A12 = tmprod(G12, U12, [1, 2, 3]);Err12 = norm(A12(:) - A2(:))/normA;ERR12(k, sample) = Err12;T12(k, sample) = t12;
        A13 = tmprod(G13, U13, [1, 2, 3]);Err13 = norm(A13(:) - A3(:))/normA;ERR13(k, sample) = Err13;T13(k, sample) = t13;
        A14 = tmprod(G14, U14, [1, 2, 3]);Err14 = norm(A14(:) - A4(:))/normA;ERR14(k, sample) = Err14;T14(k, sample) = t14;
        A15 = tmprod(G15, U15, [1, 2, 3]);Err15 = norm(A15(:) - A5(:))/normA;ERR15(k, sample) = Err15;T15(k, sample) = t15;
        A16 = tmprod(G16, U16, [1, 2, 3]);Err16 = norm(A16(:) - A6(:))/normA;ERR16(k, sample) = Err16;T16(k, sample) = t16;
        ML11(k, :) = ML11(k, :) + mult_rank11;
        ML12(k, :) = ML12(k, :) + mult_rank12;
        ML13(k, :) = ML13(k, :) + mult_rank13;
        ML14(k, :) = ML14(k, :) + mult_rank14;
        ML15(k, :) = ML15(k, :) + mult_rank15;
        ML16(k, :) = ML16(k, :) + mult_rank16;
    end
end
format short;
ML = [ML11, ML12, ML13, ML14, ML15, ML16]/10
format short e;
T = [sum(T11, 2), sum(T12, 2), sum(T13, 2), sum(T14, 2), sum(T15, 2), sum(T16, 2)]/10
ERR = [sum(ERR11, 2), sum(ERR12, 2), sum(ERR13, 2), sum(ERR14, 2), sum(ERR15, 2), sum(ERR16, 2)]/10
end