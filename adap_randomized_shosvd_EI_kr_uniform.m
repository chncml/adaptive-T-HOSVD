function [G, U, mult_rank, time] = adap_randomized_shosvd_EI_kr_uniform(A, relerr, b, P)
%% input
time = tic;
m = size(A);n = length(m);U = cell(1, n);
mult_rank = zeros(1, n);
relerr = relerr/sqrt(n);normA = norm(A(:));
relerr = relerr * normA;relerr = relerr^2;
for i = 1 : n
    B = permute(A, [i, (i - 1) : -1 : 1, (i + 1) : n]);
    % A1 = reshape(A, m(i), []);
    [Q, ~] = randQB_EI_auto_kr_uniform(B, relerr, b, P);
    mult_rank(i) = size(Q, 2);m(i) = mult_rank(i);
    U{i} = Q;
    A = tmprod(A, Q, i, 'T');
end
G = A;
end