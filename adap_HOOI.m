function [G, U, multi_rank] = adap_HOOI(A, U0, relerr)
%% A: input tensor; U0: initial gauss for factor matrices;
% relerr: error tolerance
m = size(A);n = length(m);normA = norm(A(:));
G = lmlragen(U0, A);%iter = 0;
normG = norm(G(:));
while normG > sqrt(1-relerr) * normA
    for i = 1 : n
        Utemp = U0;Utemp{i} = eye(m(i));
        B = lmlragen(Utemp, A);
        B = permute(A,[i, 1:i-1, i+1:n]);
        Bmat = reshape(B, m(i), []);
        [U, D, ~] = svd(Bmat, 0);
        d = diag(D);d = d(end:-1:1);
        d = cumsum(d.^2);
        [~, ind] = find(d<=norm(B(:))^2-(1-relerr) * normA^2);
        multi_rank(i) = m(i) - max(ind);
        U0{i} = U(:, 1 : multi_rank(i));
    end
    normG = norm(D(1 : multi_rank(i), 1 : multi_rank(i)), 'fro');
end
U = U0;
G = lmlragen(U, A);
end