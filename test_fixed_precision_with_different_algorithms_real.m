function test_fixed_precision_with_different_algorithms_real
%% comparison on tensors from real datasets
clear;clc;

relerr1 = [0.5, 0.1, 0.05];

% load('dataYaleB.mat');A = X;normA = norm(A(:));
% 
% 
% T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);
% T8 = zeros(3,1);T9 = zeros(3,1);T10 = zeros(3,1);T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);
% ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);
% ERR8 = zeros(3,1);ERR9 = zeros(3,1);ERR10 = zeros(3,1);ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);
% ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);
% ML8 = zeros(3,3);ML9 = zeros(3,3);ML10 = zeros(3,3);ML11 = zeros(3,3);ML12 =zeros(3,3);ML13 =zeros(3,3);
% 
% for k = 1 : 3
%     relerr = relerr1(k);
%     for sample = 1 : 10
%         t1 = tic;[T, mult_rank] = greedy_hosvd(A, relerr);t1 = toc(t1);
%         A1 = tmprod(T.core, T.U, [1 : 3]);
%         ERR1(k) = ERR1(k) + norm(A1(:) - A(:))/normA;ML1(k,:) = ML1(k,:)+mult_rank;
%         T1(k) = T1(k) + t1;
% 
%         t2 = tic;[U, S] = mlsvd(A, relerr, 0);t2 = toc(t2);
%         A2 = tmprod(S, U, [1:3]);
%         ERR2(k) = ERR2(k) + norm(A2(:) - A(:))/normA;ML2(k,:) = ML2(k,:)+size(S);
%         T2(k) = T2(k) + t2;
% 
%         t3 = tic;[T, ranks, ~] = rank_ada_hooi(A, relerr);t3 = toc(t3);
%         A3 = tmprod(T.core, T.U, [1 : 3]);
%         ERR3(k) = ERR3(k) + norm(A3(:) - A(:))/normA;ML3(k,:) = ML3(k,:)+ranks;
%         T3(k) = T3(k) + t3;
% 
%         t4 = tic;[core_new, F] = rtsms(A, 4, [1:3], relerr);t4 = toc(t4);
%         A4 = tmprod(core_new, F, [1:3]);
%         ERR4(k) = ERR4(k) + norm(A4(:) - A(:))/normA;ML4(k,:) = ML4(k,:)+size(core_new);
%         T4(k) = T4(k) + t4;
% 
%         t5 = tic;[G5, U5, mult_rank5] = adap_randomized_hosvd_rankone_gaussian(A, relerr, 10);t5 = toc(t5);
%         A5 = tmprod(G5, U5, [1, 2, 3]);
%         ERR5(k) = ERR5(k) + norm(A5(:) - A(:))/normA;ML5(k,:) = ML5(k,:)+mult_rank5;
%         T5(k) = T5(k) + t5;
% 
%         t6 = tic;[G6, U6, mult_rank6] = adap_randomized_hosvd_rankone_kr(A, relerr, 10);t6 = toc(t6);
%         A6 = tmprod(G6, U6, [1, 2, 3]);
%         ERR6(k) = ERR6(k) + norm(A6(:) - A(:))/normA;ML6(k,:) = ML6(k,:)+mult_rank6;
%         T6(k) = T6(k) + t6;
% 
%         t7 = tic;[G7, U7, mult_rank7] = adap_randomized_hosvd_EI_kr_gaussian(A, relerr, 60, 1);t7 = toc(t7);
%         A7 = tmprod(G7, U7, [1, 2, 3]);
%         ERR7(k) = ERR7(k) + norm(A7(:) - A(:))/normA;ML7(k,:) = ML7(k,:)+mult_rank7;
%         T7(k) = T7(k) + t7;
% 
%         t8 = tic;[G8, U8, mult_rank8] = adap_randomized_hosvd_EI_kr_bernoulli(A, relerr, 60, 1);t8 = toc(t8);
%         A8 = tmprod(G8, U8, [1, 2, 3]);
%         ERR8(k) = ERR8(k) + norm(A8(:) - A(:))/normA;ML8(k,:) = ML8(k,:)+mult_rank8;
%         T8(k) = T8(k) + t8;
% 
%         t9 = tic;[U, S] = mlsvd(A, relerr);t9 = toc(t9);
%         A9 = tmprod(S, U, [1:3]);
%         ERR9(k) = ERR9(k) + norm(A9(:) - A(:))/normA;ML9(k,:) = ML9(k,:)+size(S);
%         T9(k) = T9(k) + t9;
% 
%         t10 = tic;[G10, U10, mult_rank10] = adap_randomized_shosvd_rankone_gaussian(A, relerr, 10);t10 = toc(t10);
%         A10 = tmprod(G10, U10, [1, 2, 3]);
%         ERR10(k) = ERR10(k) + norm(A10(:) - A(:))/normA;ML10(k,:) = ML10(k,:)+mult_rank10;
%         T10(k) = T10(k) + t10;
% 
%         t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_rankone_kr(A, relerr, 10);t11 = toc(t11);
%         A11 = tmprod(G11, U11, [1, 2, 3]);
%         ERR11(k) = ERR11(k) + norm(A11(:) - A(:))/normA;ML11(k,:) = ML11(k,:)+mult_rank11;
%         T11(k) = T11(k) + t11;
% 
%         t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t12 = toc(t12);
%         A12 = tmprod(G12, U12, [1, 2, 3]);
%         ERR12(k) = ERR12(k) + norm(A12(:) - A(:))/normA;ML12(k,:) = ML12(k,:)+mult_rank12;
%         T12(k) = T12(k) + t12;
% 
%         t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t13 = toc(t13);
%         A13 = tmprod(G13, U13, [1, 2, 3]);
%         ERR13(k) = ERR13(k) + norm(A13(:) - A(:))/normA;ML13(k,:) = ML13(k,:)+mult_rank13;
%         T13(k) = T13(k) + t13;
%     end
% end
% 
% format short e;
% [ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8, ML9, ML10, ML11, ML12, ML13]/10
% [ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8, ERR9, ERR10, ERR11, ERR12, ERR13]/10
% [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13]/10
% 
% 
% load('washington.mat');
% A = washington_DC_mall_double_data;normA = norm(A(:));
% 
% 
% T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);
% T8 = zeros(3,1);T9 = zeros(3,1);T10 = zeros(3,1);T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);
% ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);
% ERR8 = zeros(3,1);ERR9 = zeros(3,1);ERR10 = zeros(3,1);ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);
% ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);
% ML8 = zeros(3,3);ML9 = zeros(3,3);ML10 = zeros(3,3);ML11 = zeros(3,3);ML12 =zeros(3,3);ML13 =zeros(3,3);
% 
% for k = 1 : 3
%     relerr = relerr1(k);
%     for sample = 1 : 10
%         t1 = tic;[T, mult_rank] = greedy_hosvd(A, relerr);t1 = toc(t1);
%         A1 = tmprod(T.core, T.U, [1 : 3]);
%         ERR1(k) = ERR1(k) + norm(A1(:) - A(:))/normA;ML1(k,:) = ML1(k,:)+mult_rank;
%         T1(k) = T1(k) + t1;
% 
%         t2 = tic;[U, S] = mlsvd(A, relerr, 0);t2 = toc(t2);
%         A2 = tmprod(S, U, [1:3]);
%         ERR2(k) = ERR2(k) + norm(A2(:) - A(:))/normA;ML2(k,:) = ML2(k,:)+size(S);
%         T2(k) = T2(k) + t2;
% 
%         t3 = tic;[T, ranks, ~] = rank_ada_hooi(A, relerr);t3 = toc(t3);
%         A3 = tmprod(T.core, T.U, [1 : 3]);
%         ERR3(k) = ERR3(k) + norm(A3(:) - A(:))/normA;ML3(k,:) = ML3(k,:)+ranks;
%         T3(k) = T3(k) + t3;
% 
%         t4 = tic;[core_new, F] = rtsms(A, 4, [1:3], relerr);t4 = toc(t4);
%         A4 = tmprod(core_new, F, [1:3]);
%         ERR4(k) = ERR4(k) + norm(A4(:) - A(:))/normA;ML4(k,:) = ML4(k,:)+size(core_new);
%         T4(k) = T4(k) + t4;
% 
%         t5 = tic;[G5, U5, mult_rank5] = adap_randomized_hosvd_rankone_gaussian(A, relerr, 10);t5 = toc(t5);
%         A5 = tmprod(G5, U5, [1, 2, 3]);
%         ERR5(k) = ERR5(k) + norm(A5(:) - A(:))/normA;ML5(k,:) = ML5(k,:)+mult_rank5;
%         T5(k) = T5(k) + t5;
% 
%         t6 = tic;[G6, U6, mult_rank6] = adap_randomized_hosvd_rankone_kr(A, relerr, 10);t6 = toc(t6);
%         A6 = tmprod(G6, U6, [1, 2, 3]);
%         ERR6(k) = ERR6(k) + norm(A6(:) - A(:))/normA;ML6(k,:) = ML6(k,:)+mult_rank6;
%         T6(k) = T6(k) + t6;
% 
%         t7 = tic;[G7, U7, mult_rank7] = adap_randomized_hosvd_EI_kr_gaussian(A, relerr, 60, 1);t7 = toc(t7);
%         A7 = tmprod(G7, U7, [1, 2, 3]);
%         ERR7(k) = ERR7(k) + norm(A7(:) - A(:))/normA;ML7(k,:) = ML7(k,:)+mult_rank7;
%         T7(k) = T7(k) + t7;
% 
%         t8 = tic;[G8, U8, mult_rank8] = adap_randomized_hosvd_EI_kr_bernoulli(A, relerr, 60, 1);t8 = toc(t8);
%         A8 = tmprod(G8, U8, [1, 2, 3]);
%         ERR8(k) = ERR8(k) + norm(A8(:) - A(:))/normA;ML8(k,:) = ML8(k,:)+mult_rank8;
%         T8(k) = T8(k) + t8;
% 
%         t9 = tic;[U, S] = mlsvd(A, relerr);t9 = toc(t9);
%         A9 = tmprod(S, U, [1:3]);
%         ERR9(k) = ERR9(k) + norm(A9(:) - A(:))/normA;ML9(k,:) = ML9(k,:)+size(S);
%         T9(k) = T9(k) + t9;
% 
%         t10 = tic;[G10, U10, mult_rank10] = adap_randomized_shosvd_rankone_gaussian(A, relerr, 10);t10 = toc(t10);
%         A10 = tmprod(G10, U10, [1, 2, 3]);
%         ERR10(k) = ERR10(k) + norm(A10(:) - A(:))/normA;ML10(k,:) = ML10(k,:)+mult_rank10;
%         T10(k) = T10(k) + t10;
% 
%         t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_rankone_kr(A, relerr, 10);t11 = toc(t11);
%         A11 = tmprod(G11, U11, [1, 2, 3]);
%         ERR11(k) = ERR11(k) + norm(A11(:) - A(:))/normA;ML11(k,:) = ML11(k,:)+mult_rank11;
%         T11(k) = T11(k) + t11;
% 
%         t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t12 = toc(t12);
%         A12 = tmprod(G12, U12, [1, 2, 3]);
%         ERR12(k) = ERR12(k) + norm(A12(:) - A(:))/normA;ML12(k,:) = ML12(k,:)+mult_rank12;
%         T12(k) = T12(k) + t12;
% 
%         t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t13 = toc(t13);
%         A13 = tmprod(G13, U13, [1, 2, 3]);
%         ERR13(k) = ERR13(k) + norm(A13(:) - A(:))/normA;ML13(k,:) = ML13(k,:)+mult_rank13;
%         T13(k) = T13(k) + t13;
%     end
% end
% 
% format short e;
% [ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8, ML9, ML10, ML11, ML12, ML13]/10
% [ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8, ERR9, ERR10, ERR11, ERR12, ERR13]/10
% [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13]/10


load('coil100A.mat');load('coil100B.mat');
B(:, :, :, 1 : 50, :) = A1;B(:, :, :, 51 : 100, :) = A2;
A = reshape(B, [1024, 1152, 300]);
normA = norm(A(:));


T1 = zeros(3,1);T2 = zeros(3,1);T3 = zeros(3,1);T4 = zeros(3,1);T5 = zeros(3,1);T6 = zeros(3,1);T7 = zeros(3,1);
T8 = zeros(3,1);T9 = zeros(3,1);T10 = zeros(3,1);T11 = zeros(3,1);T12 = zeros(3,1);T13 = zeros(3,1);
ERR1 = zeros(3,1);ERR2 = zeros(3,1);ERR3 = zeros(3,1);ERR4 = zeros(3,1);ERR5 = zeros(3,1);ERR6 = zeros(3,1);ERR7 = zeros(3,1);
ERR8 = zeros(3,1);ERR9 = zeros(3,1);ERR10 = zeros(3,1);ERR11 = zeros(3,1);ERR12 = zeros(3,1);ERR13 = zeros(3,1);
ML1 = zeros(3,3);ML2 = zeros(3,3);ML3 = zeros(3,3);ML4 = zeros(3,3);ML5 = zeros(3,3);ML6 = zeros(3,3);ML7 = zeros(3,3);
ML8 = zeros(3,3);ML9 = zeros(3,3);ML10 = zeros(3,3);ML11 = zeros(3,3);ML12 =zeros(3,3);ML13 =zeros(3,3);

for k = 1 : 3
    relerr = relerr1(k);
    for sample = 1 : 10
        sample
        t1 = tic;[T, mult_rank] = greedy_hosvd(A, relerr);t1 = toc(t1)
        A1 = tmprod(T.core, T.U, [1 : 3]);
        ERR1(k) = ERR1(k) + norm(A1(:) - A(:))/normA;ML1(k,:) = ML1(k,:)+mult_rank;
        T1(k) = T1(k) + t1;
        
        t2 = tic;[U, S] = mlsvd(A, relerr, 0);t2 = toc(t2)
        A2 = tmprod(S, U, [1:3]);
        ERR2(k) = ERR2(k) + norm(A2(:) - A(:))/normA;ML2(k,:) = ML2(k,:)+size(S);
        T2(k) = T2(k) + t2;
       
        t3 = tic;[T, ranks, ~] = rank_ada_hooi(A, relerr);t3 = toc(t3)
        A3 = tmprod(T.core, T.U, [1 : 3]);
        ERR3(k) = ERR3(k) + norm(A3(:) - A(:))/normA;ML3(k,:) = ML3(k,:)+ranks;
        T3(k) = T3(k) + t3;

        t4 = tic;[core_new, F] = rtsms(A, 4, [1:3], relerr);t4 = toc(t4)
        A4 = tmprod(core_new, F, [1:3]);
        ERR4(k) = ERR4(k) + norm(A4(:) - A(:))/normA;ML4(k,:) = ML4(k,:)+size(core_new);
        T4(k) = T4(k) + t4;

        t5 = tic;[G5, U5, mult_rank5] = adap_randomized_hosvd_rankone_gaussian(A, relerr, 10);t5 = toc(t5)
        A5 = tmprod(G5, U5, [1, 2, 3]);
        ERR5(k) = ERR5(k) + norm(A5(:) - A(:))/normA;ML5(k,:) = ML5(k,:)+mult_rank5;
        T5(k) = T5(k) + t5;

        t6 = tic;[G6, U6, mult_rank6] = adap_randomized_hosvd_rankone_kr(A, relerr, 10);t6 = toc(t6)
        A6 = tmprod(G6, U6, [1, 2, 3]);
        ERR6(k) = ERR6(k) + norm(A6(:) - A(:))/normA;ML6(k,:) = ML6(k,:)+mult_rank6;
        T6(k) = T6(k) + t6;

        t7 = tic;[G7, U7, mult_rank7] = adap_randomized_hosvd_EI_kr_gaussian(A, relerr, 60, 1);t7 = toc(t7)
        A7 = tmprod(G7, U7, [1, 2, 3]);
        ERR7(k) = ERR7(k) + norm(A7(:) - A(:))/normA;ML7(k,:) = ML7(k,:)+mult_rank7;
        T7(k) = T7(k) + t7;

        t8 = tic;[G8, U8, mult_rank8] = adap_randomized_hosvd_EI_kr_bernoulli(A, relerr, 60, 1);t8 = toc(t8)
        A8 = tmprod(G8, U8, [1, 2, 3]);
        ERR8(k) = ERR8(k) + norm(A8(:) - A(:))/normA;ML8(k,:) = ML8(k,:)+mult_rank8;
        T8(k) = T8(k) + t8;

        t9 = tic;[U, S] = mlsvd(A, relerr);t9 = toc(t9)
        A9 = tmprod(S, U, [1:3]);
        ERR9(k) = ERR9(k) + norm(A9(:) - A(:))/normA;ML9(k,:) = ML9(k,:)+size(S);
        T9(k) = T9(k) + t9;

        t10 = tic;[G10, U10, mult_rank10] = adap_randomized_shosvd_rankone_gaussian(A, relerr, 10);t10 = toc(t10)
        A10 = tmprod(G10, U10, [1, 2, 3]);
        ERR10(k) = ERR10(k) + norm(A10(:) - A(:))/normA;ML10(k,:) = ML10(k,:)+mult_rank10;
        T10(k) = T10(k) + t10;

        t11 = tic;[G11, U11, mult_rank11] = adap_randomized_shosvd_rankone_kr(A, relerr, 10);t11 = toc(t11)
        A11 = tmprod(G11, U11, [1, 2, 3]);
        ERR11(k) = ERR11(k) + norm(A11(:) - A(:))/normA;ML11(k,:) = ML11(k,:)+mult_rank11;
        T11(k) = T11(k) + t11;

        t12 = tic;[G12, U12, mult_rank12] = adap_randomized_shosvd_EI_kr_gaussian(A, relerr, 60, 1);t12 = toc(t12)
        A12 = tmprod(G12, U12, [1, 2, 3]);
        ERR12(k) = ERR12(k) + norm(A12(:) - A(:))/normA;ML12(k,:) = ML12(k,:)+mult_rank12;
        T12(k) = T12(k) + t12;

        t13 = tic;[G13, U13, mult_rank13] = adap_randomized_shosvd_EI_kr_bernoulli(A, relerr, 60, 1);t13 = toc(t13)
        A13 = tmprod(G13, U13, [1, 2, 3]);
        ERR13(k) = ERR13(k) + norm(A13(:) - A(:))/normA;ML13(k,:) = ML13(k,:)+mult_rank13;
        T13(k) = T13(k) + t13;
    end
end

format short e;
[ML1, ML2, ML3, ML4, ML5, ML6, ML7, ML8, ML9, ML10, ML11, ML12, ML13]/10
[ERR1, ERR2, ERR3, ERR4, ERR5, ERR6, ERR7, ERR8, ERR9, ERR10, ERR11, ERR12, ERR13]/10
[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13]/10

end