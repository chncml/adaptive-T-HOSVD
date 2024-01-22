function [T,ranks,iter] = rank_ada_hooi(A,epsilon,varargin)

% compute the truncated Tucker decomposition with a given tolerance
% by Chuanfu Xiao

sizeA = size(A);
N = length(sizeA);
normA = frob(A);

params = inputParser;
params.addParameter('dimorder',1:N);
params.addParameter('init','rand',@(x) (iscell(x) || ismember(x,{'rand'})));
params.addParameter('delta_r',ones(1,N));
params.addParameter('tol',1.0e-4);
params.addParameter('maxiters',500);
params.parse(varargin{:});

dimorder = params.Results.dimorder;
init = params.Results.init;
delta_r = params.Results.delta_r;
tol = params.Results.tol;
maxiters = params.Results.maxiters;

if iscell(init)
    ranks = zeros(1,N);
    U = init;
    for n = 1:N
        ranks(n) = size(U{n},2);
    end
    G = tmprod(A,U,1:N,'T');
elseif strcmp(init,'rand')
    ranks = 2*ones(1,N);
    U = cell(N,1);
    for n = 1:N
        Q = randn(sizeA(n),ranks(n));
        [Q,~] = qr(Q,0);
        U{n} = Q;
    end
    G = tmprod(A,U,1:N,'T');
end

relres = sqrt(1-frob(G)^2/normA^2);

% fprintf('==========================Rank-adaptive HOOI==========================\n');
% fprintf('Tucker-rank is %d\n', ranks);
% fprintf('Approximation error is %.16f\n', relres);

for k = 1:maxiters
    for n = dimorder
        mode = [1:n-1 n+1:N];
        B = tmprod(A,U(mode),mode,'T');
        CHECK_ERR = frob(B)^2 - (1-epsilon^2)*normA^2;
        B_n = tens2mat(B,n);
        if CHECK_ERR <= eps
            r = min(ranks(n) + delta_r(n),sizeA(n));
            [Q,~,~] = svds(B_n,r);
            U{n} = Q;
            ranks(n) = r;
        elseif CHECK_ERR > eps
            midepsilon = sqrt(CHECK_ERR);
            [Q,S,~] = svd(B_n,'econ');
            s_sum = 0;
            for i = length(diag(S)):-1:1
                s_sum = s_sum + S(i,i)^2;
                if sqrt(s_sum) >= midepsilon
                    U{n} = Q(:,1:i);
                    ranks(n) = i;
                    break;
                end
            end
        end
    end

    G = tmprod(B,U{n},n,'T');

    relres_1 = sqrt(1 - frob(G)^2/normA^2);
    CHECK_RES = abs(relres_1 - relres);
    relres = relres_1;

%     fprintf('=========================%d-th iteration========================\n',k);
%     fprintf('Tucker-rank is %d\n', ranks);
%     fprintf('Approximation error is %.16f\n', relres);

    if relres <= epsilon && CHECK_RES <= tol
        break;
    end
end

T.core = G;
T.U = U;
iter = k;

end