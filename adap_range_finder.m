function [Q, mu, time] = adap_range_finder(A, relerr, OV)
%% adaptive randomized range finder
time = tic;
[m, n] = size(A);
Q = zeros(m, 0);mu = 0;
G = randn(n, OV);Y = A * G;
% relerr = relerr * norm(A(:))/(10 * sqrt(2/pi));
tempmax = sqrt(sum(Y.^2));
while max(tempmax) > relerr
    mu = mu + 1;
    tempvector = Y(:, 1) - Q * (Q' * Y(:, 1));
    q = tempvector/norm(tempvector);
    Q = [Q, q];
    y = A * randn(n, 1);y = y - Q * (Q' * y);
    tempmatrix = Y(:, 2 : OV);
    tempmatrix = tempmatrix - q * (q' * tempmatrix);
    Y = [tempmatrix, y];
    tempmax = sqrt(sum(Y.^2));
end
time = toc(time);
end