function test_fixed_precision_with_fixed_changed
%% comparison on fixed and changed
clc;clear;

%% Yale
relerr1 = [0.5, 0.1, 0.05];
load('dataYaleB.mat');A = X;normA = norm(A(:));

T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);
T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);T8 = zeros(3,1);
ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);
ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);ERR8 = zeros(3,1);
ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);
ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);ML8 = zeros(3,3);

for k = 1 : 3
    relerr = relerr1(k);
    for sample = 1 : 10
        t1 = tic;[G1, U1, mult_rank1] = adap_randomized_hosvd_EI_gaussian(A, relerr, 60, 1);t1 = toc(t1);
        t2 = tic;[G2, U2, mult_rank2] = adap_randomized_hosvd_EI_gaussian_fix(A, relerr, 60, 1);t2 = toc(t2);
        t3 = tic;[G3, U3, mult_rank3] = adap_randomized_hosvd_EI_bernoulli(A, relerr, 60, 1);t3 = toc(t3);
        t4 = tic;[G4, U4, mult_rank4] = adap_randomized_hosvd_EI_bernoulli_fix(A, relerr, 60, 1);t4 = toc(t4);
        t5 = tic;[G5, U5, mult_rank5] = adap_randomized_hosvd_EI_kr_gaussian(A, relerr, 60, 1);t5 = toc(t5);
        t6 = tic;[G6, U6, mult_rank6] = adap_randomized_hosvd_EI_kr_gaussian_fix(A, relerr, 60, 1);t6 = toc(t6);
        t7 = tic;[G7, U7, mult_rank7] = adap_randomized_hosvd_EI_kr_bernoulli(A, relerr, 60, 1);t7 = toc(t7);
        t8 = tic;[G8, U8, mult_rank8] = adap_randomized_hosvd_EI_kr_bernoulli_fix(A, relerr, 60, 1);t8 = toc(t8);
        A1 = tmprod(G1, U1, [1:3]);Err1 = norm(A1(:) - A(:))/normA;
        A2 = tmprod(G2, U2, [1:3]);Err2 = norm(A2(:) - A(:))/normA;
        A3 = tmprod(G3, U3, [1:3]);Err3 = norm(A3(:) - A(:))/normA;
        A4 = tmprod(G4, U4, [1:3]);Err4 = norm(A4(:) - A(:))/normA;
        A5 = tmprod(G5, U5, [1:3]);Err5 = norm(A5(:) - A(:))/normA;
        A6 = tmprod(G6, U6, [1:3]);Err6 = norm(A6(:) - A(:))/normA;
        A7 = tmprod(G7, U7, [1:3]);Err7 = norm(A7(:) - A(:))/normA;
        A8 = tmprod(G8, U8, [1:3]);Err8 = norm(A8(:) - A(:))/normA;
        ERR1(k) = ERR1(k) + Err1;ML1(k,:) = ML1(k,:)+mult_rank1;T1(k) = T1(k) + t1;
        ERR2(k) = ERR2(k) + Err2;ML2(k,:) = ML2(k,:)+mult_rank2;T2(k) = T2(k) + t2;
        ERR3(k) = ERR3(k) + Err3;ML3(k,:) = ML3(k,:)+mult_rank3;T3(k) = T3(k) + t3;
        ERR4(k) = ERR4(k) + Err4;ML4(k,:) = ML4(k,:)+mult_rank4;T4(k) = T4(k) + t4;
        ERR5(k) = ERR5(k) + Err5;ML5(k,:) = ML5(k,:)+mult_rank5;T5(k) = T5(k) + t5;
        ERR6(k) = ERR6(k) + Err6;ML6(k,:) = ML6(k,:)+mult_rank6;T6(k) = T6(k) + t6;
        ERR7(k) = ERR7(k) + Err7;ML7(k,:) = ML7(k,:)+mult_rank7;T7(k) = T7(k) + t7;
        ERR8(k) = ERR8(k) + Err8;ML8(k,:) = ML8(k,:)+mult_rank8;T8(k) = T8(k) + t8;
    end
end

format short e;
[ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8]/10
[ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8]/10
[T1, T2, T3, T4, T5, T6, T7, T8]/10


T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);
T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);T8 = zeros(3,1);
ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);
ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);ERR8 = zeros(3,1);
ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);
ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);ML8 = zeros(3,3);

for k = 1 : 3
    relerr = relerr1(k);
    for sample = 1 : 10
        t1 = tic;[G1, U1, mult_rank1] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 1);t1 = toc(t1);
        t2 = tic;[G2, U2, mult_rank2] = adap_randomized_shosvd_EI_gaussian_fix(A, relerr, 60, 1);t2 = toc(t2);
        t3 = tic;[G3, U3, mult_rank3] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 1);t3 = toc(t3);
        t4 = tic;[G4, U4, mult_rank4] = adap_randomized_shosvd_EI_bernoulli_fix(A, relerr, 60, 1);t4 = toc(t4);
        t5 = tic;[G5, U5, mult_rank5] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t5 = toc(t5);
        t6 = tic;[G6, U6, mult_rank6] = adap_randomized_shosvd_EI_kr_gaussian_fix(A, relerr, 60, 1);t6 = toc(t6);
        t7 = tic;[G7, U7, mult_rank7] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t7 = toc(t7);
        t8 = tic;[G8, U8, mult_rank8] = adap_randomized_shosvd_EI_kr_bernoulli_fix(A, relerr, 60, 1);t8 = toc(t8);
        A1 = tmprod(G1, U1, [1:3]);Err1 = norm(A1(:) - A(:))/normA;
        A2 = tmprod(G2, U2, [1:3]);Err2 = norm(A2(:) - A(:))/normA;
        A3 = tmprod(G3, U3, [1:3]);Err3 = norm(A3(:) - A(:))/normA;
        A4 = tmprod(G4, U4, [1:3]);Err4 = norm(A4(:) - A(:))/normA;
        A5 = tmprod(G5, U5, [1:3]);Err5 = norm(A5(:) - A(:))/normA;
        A6 = tmprod(G6, U6, [1:3]);Err6 = norm(A6(:) - A(:))/normA;
        A7 = tmprod(G7, U7, [1:3]);Err7 = norm(A7(:) - A(:))/normA;
        A8 = tmprod(G8, U8, [1:3]);Err8 = norm(A8(:) - A(:))/normA;
        ERR1(k) = ERR1(k) + Err1;ML1(k,:) = ML1(k,:)+mult_rank1;T1(k) = T1(k) + t1;
        ERR2(k) = ERR2(k) + Err2;ML2(k,:) = ML2(k,:)+mult_rank2;T2(k) = T2(k) + t2;
        ERR3(k) = ERR3(k) + Err3;ML3(k,:) = ML3(k,:)+mult_rank3;T3(k) = T3(k) + t3;
        ERR4(k) = ERR4(k) + Err4;ML4(k,:) = ML4(k,:)+mult_rank4;T4(k) = T4(k) + t4;
        ERR5(k) = ERR5(k) + Err5;ML5(k,:) = ML5(k,:)+mult_rank5;T5(k) = T5(k) + t5;
        ERR6(k) = ERR6(k) + Err6;ML6(k,:) = ML6(k,:)+mult_rank6;T6(k) = T6(k) + t6;
        ERR7(k) = ERR7(k) + Err7;ML7(k,:) = ML7(k,:)+mult_rank7;T7(k) = T7(k) + t7;
        ERR8(k) = ERR8(k) + Err8;ML8(k,:) = ML8(k,:)+mult_rank8;T8(k) = T8(k) + t8;
    end
end

format short e;
[ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8]/10
[ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8]/10
[T1, T2, T3, T4, T5, T6, T7, T8]/10

%% slow decay
n = 800;b1 = randn(n);[b1, ~] = qr(b1);b2 = randn(n);[b2, ~] = qr(b2);
b3 = randn(n);[b3, ~] = qr(b3);

relerr1 = [1e-1, 1e-2, 1e-3];


v = [1 : n];
v = v.^2;v = 1./v;

A = tendiag(v, [n, n, n]);
A = ttm(A, {b1, b2, b3}, [1, 2, 3]);
A = double(A);normA = norm(A(:));

T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);
T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);T8 = zeros(3,1);
ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);
ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);ERR8 = zeros(3,1);
ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);
ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);ML8 = zeros(3,3);

for k = 1 : 3
    relerr = relerr1(k);
    for sample = 1 : 10
        t1 = tic;[G1, U1, mult_rank1] = adap_randomized_hosvd_EI_gaussian(A, relerr, 60, 1);t1 = toc(t1);
        t2 = tic;[G2, U2, mult_rank2] = adap_randomized_hosvd_EI_gaussian_fix(A, relerr, 60, 1);t2 = toc(t2);
        t3 = tic;[G3, U3, mult_rank3] = adap_randomized_hosvd_EI_bernoulli(A, relerr, 60, 1);t3 = toc(t3);
        t4 = tic;[G4, U4, mult_rank4] = adap_randomized_hosvd_EI_bernoulli_fix(A, relerr, 60, 1);t4 = toc(t4);
        t5 = tic;[G5, U5, mult_rank5] = adap_randomized_hosvd_EI_kr_gaussian(A, relerr, 60, 1);t5 = toc(t5);
        t6 = tic;[G6, U6, mult_rank6] = adap_randomized_hosvd_EI_kr_gaussian_fix(A, relerr, 60, 1);t6 = toc(t6);
        t7 = tic;[G7, U7, mult_rank7] = adap_randomized_hosvd_EI_kr_bernoulli(A, relerr, 60, 1);t7 = toc(t7);
        t8 = tic;[G8, U8, mult_rank8] = adap_randomized_hosvd_EI_kr_bernoulli_fix(A, relerr, 60, 1);t8 = toc(t8);
        A1 = tmprod(G1, U1, [1:3]);Err1 = norm(A1(:) - A(:))/normA;
        A2 = tmprod(G2, U2, [1:3]);Err2 = norm(A2(:) - A(:))/normA;
        A3 = tmprod(G3, U3, [1:3]);Err3 = norm(A3(:) - A(:))/normA;
        A4 = tmprod(G4, U4, [1:3]);Err4 = norm(A4(:) - A(:))/normA;
        A5 = tmprod(G5, U5, [1:3]);Err5 = norm(A5(:) - A(:))/normA;
        A6 = tmprod(G6, U6, [1:3]);Err6 = norm(A6(:) - A(:))/normA;
        A7 = tmprod(G7, U7, [1:3]);Err7 = norm(A7(:) - A(:))/normA;
        A8 = tmprod(G8, U8, [1:3]);Err8 = norm(A8(:) - A(:))/normA;
        ERR1(k) = ERR1(k) + Err1;ML1(k,:) = ML1(k,:)+mult_rank1;T1(k) = T1(k) + t1;
        ERR2(k) = ERR2(k) + Err2;ML2(k,:) = ML2(k,:)+mult_rank2;T2(k) = T2(k) + t2;
        ERR3(k) = ERR3(k) + Err3;ML3(k,:) = ML3(k,:)+mult_rank3;T3(k) = T3(k) + t3;
        ERR4(k) = ERR4(k) + Err4;ML4(k,:) = ML4(k,:)+mult_rank4;T4(k) = T4(k) + t4;
        ERR5(k) = ERR5(k) + Err5;ML5(k,:) = ML5(k,:)+mult_rank5;T5(k) = T5(k) + t5;
        ERR6(k) = ERR6(k) + Err6;ML6(k,:) = ML6(k,:)+mult_rank6;T6(k) = T6(k) + t6;
        ERR7(k) = ERR7(k) + Err7;ML7(k,:) = ML7(k,:)+mult_rank7;T7(k) = T7(k) + t7;
        ERR8(k) = ERR8(k) + Err8;ML8(k,:) = ML8(k,:)+mult_rank8;T8(k) = T8(k) + t8;
    end
end

format short e;
[ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8]/10
[ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8]/10
[T1, T2, T3, T4, T5, T6, T7, T8]/10


T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);
T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);T8 = zeros(3,1);
ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);
ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);ERR8 = zeros(3,1);
ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);
ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);ML8 = zeros(3,3);

for k = 1 : 3
    relerr = relerr1(k);
    for sample = 1 : 10
        t1 = tic;[G1, U1, mult_rank1] = adap_randomized_shosvd_EI_gaussian(A, relerr, 60, 1);t1 = toc(t1);
        t2 = tic;[G2, U2, mult_rank2] = adap_randomized_shosvd_EI_gaussian_fix(A, relerr, 60, 1);t2 = toc(t2);
        t3 = tic;[G3, U3, mult_rank3] = adap_randomized_shosvd_EI_bernoulli(A, relerr, 60, 1);t3 = toc(t3);
        t4 = tic;[G4, U4, mult_rank4] = adap_randomized_shosvd_EI_bernoulli_fix(A, relerr, 60, 1);t4 = toc(t4);
        t5 = tic;[G5, U5, mult_rank5] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t5 = toc(t5);
        t6 = tic;[G6, U6, mult_rank6] = adap_randomized_shosvd_EI_kr_gaussian_fix(A, relerr, 60, 1);t6 = toc(t6);
        t7 = tic;[G7, U7, mult_rank7] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t7 = toc(t7);
        t8 = tic;[G8, U8, mult_rank8] = adap_randomized_shosvd_EI_kr_bernoulli_fix(A, relerr, 60, 1);t8 = toc(t8);
        A1 = tmprod(G1, U1, [1:3]);Err1 = norm(A1(:) - A(:))/normA;
        A2 = tmprod(G2, U2, [1:3]);Err2 = norm(A2(:) - A(:))/normA;
        A3 = tmprod(G3, U3, [1:3]);Err3 = norm(A3(:) - A(:))/normA;
        A4 = tmprod(G4, U4, [1:3]);Err4 = norm(A4(:) - A(:))/normA;
        A5 = tmprod(G5, U5, [1:3]);Err5 = norm(A5(:) - A(:))/normA;
        A6 = tmprod(G6, U6, [1:3]);Err6 = norm(A6(:) - A(:))/normA;
        A7 = tmprod(G7, U7, [1:3]);Err7 = norm(A7(:) - A(:))/normA;
        A8 = tmprod(G8, U8, [1:3]);Err8 = norm(A8(:) - A(:))/normA;
        ERR1(k) = ERR1(k) + Err1;ML1(k,:) = ML1(k,:)+mult_rank1;T1(k) = T1(k) + t1;
        ERR2(k) = ERR2(k) + Err2;ML2(k,:) = ML2(k,:)+mult_rank2;T2(k) = T2(k) + t2;
        ERR3(k) = ERR3(k) + Err3;ML3(k,:) = ML3(k,:)+mult_rank3;T3(k) = T3(k) + t3;
        ERR4(k) = ERR4(k) + Err4;ML4(k,:) = ML4(k,:)+mult_rank4;T4(k) = T4(k) + t4;
        ERR5(k) = ERR5(k) + Err5;ML5(k,:) = ML5(k,:)+mult_rank5;T5(k) = T5(k) + t5;
        ERR6(k) = ERR6(k) + Err6;ML6(k,:) = ML6(k,:)+mult_rank6;T6(k) = T6(k) + t6;
        ERR7(k) = ERR7(k) + Err7;ML7(k,:) = ML7(k,:)+mult_rank7;T7(k) = T7(k) + t7;
        ERR8(k) = ERR8(k) + Err8;ML8(k,:) = ML8(k,:)+mult_rank8;T8(k) = T8(k) + t8;
    end
end

format short e;
[ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8]/10
[ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8]/10
[T1, T2, T3, T4, T5, T6, T7, T8]/10

end