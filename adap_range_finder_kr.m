function [Q, mu, time] = adap_range_finder_kr(A, relerr, OV)
%% adaptive randomized range finder
time = tic;
mu = 0;m = size(A);
n = length(m);Q = zeros(m(1), 0);
% relerr = relerr * norm(A(:))/(10 * sqrt(2/pi));
G = ones(1, OV);
for i = 2 : n
    G = kr(G, randn(m(i), OV));
end
A = reshape(A, m(1), []);Y = A * G;
tempmax = sqrt(sum(Y.^2));
while max(tempmax) > relerr
    mu = mu + 1;
    tempvector  = Y(:, 1) - Q * (Q' * Y(:, 1));
    q = tempvector/norm(tempvector);
    Q = [Q, q];
    G = 1;
    for i = 2 : n
        G = kr(G, randn(m(i), 1));
    end
    y = A * G;y = y - Q * (Q' * y);
    tempmatrix = Y(:, 2 : OV);
    tempmatrix = tempmatrix - q * (q' * tempmatrix);
    Y = [tempmatrix, y];
    tempmax = sqrt(sum(Y.^2));
end
time = toc(time);
end