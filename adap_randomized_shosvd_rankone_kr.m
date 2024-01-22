function [G, U, mult_rank, time] = adap_randomized_shosvd_rankone_kr(A, relerr, OV)
%% input
time = tic;
m = size(A);n = length(m);U = cell(1, n);U1 = cell(1, n);
mult_rank = zeros(1, n);
relerr = relerr/sqrt(n);normA = norm(A(:));
relerr = relerr * normA;
for i = 1 : n
    A = permute(A, [i, (i - 1) : -1 : 1, (i + 1) : n]);
    A1 = reshape(A, m(i), []);
    [Q, ~] = adap_range_finder_kr(A, relerr, OV);
    mult_rank(i) = size(Q, 2);m(i) = mult_rank(i);
    U{i} = Q;
    A = reshape(Q' * A1, [m(i), m(1 : i-1), m(i + 1 : n)]);
end
G = reshape(A, [m(n), m(1 : n - 1)]);
G = permute(G, [2 : n, 1]);
time = toc(time);
end