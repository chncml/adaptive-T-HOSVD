function [G, U, mult_rank, time] = adap_randomized_hosvd_EI_kr(A, relerr, b, P)
%% input
time = tic;m = size(A);n = length(m);U = cell(1, n);
mult_rank = zeros(1, n);
relerr = relerr/sqrt(n);normA = norm(A(:));
relerr = relerr * normA;relerr = relerr^2;
for i = 1 : n
    B = permute(A, [i, 1 : i - 1, i + 1 : n]);
    [Q, ~] = randQB_EI_auto_kr(B, relerr, b, P);
    mult_rank(i) = size(Q, 2);U{i} = Q;
end
G = tmprod(A, U, [1 : n], 'T');
time = toc(time);
end