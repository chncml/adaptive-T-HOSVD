function [T,rankss] = greedy_hosvd(A,epsilon)

% compute the truncated Tucker decomposition with a given tolerance
% greedy HOSVD is based on the t-HOSVD
% based on tensorlab

sizeA = size(A);
N = length(sizeA); 
normA = frob(A);
R = zeros(1, N);

% store the singular values decomposition factors
U = cell(N,1);
S = cell(N,1);
for i = 1:N
    Mat = tens2mat(A,i,[1:i-1 i+1:N]);
    [u,s,~] = svd(Mat,'econ');
    U{i} = u;
    S{i} = diag(s);
end    

% approximation error
CHECKERR = 0.0;
for i = 1:N
    s = S{i};
    L = length(s);
    for l = R(i)+1:L
        CHECKERR = CHECKERR + s(l)^2;
    end
end

while CHECKERR > (epsilon*normA)^2      
    if R == zeros(N,1)
        R = R + 1;
        test_svs = zeros(N,1);
        for i = 1:N
            s = S{i};
            if R(i) < length(s)
                test_svs(i) = s(R(i)+1);
            else
                test_svs(i) = 0.0;
            end
            CHECKERR = CHECKERR - s(1)^2;
        end
    else
        % determine the increased index
        [~,index] = sort(test_svs,'descend');
        my_index = index(1);

        % update rank
        R(my_index) = R(my_index) + 1;

        % update approximation
        CHECKERR = CHECKERR - test_svs(my_index)^2;

        % update test_svs
        s = S{my_index};
        if R(my_index) < length(s)
            test_svs(my_index) = s(R(my_index)+1);
        else
            test_svs(my_index) = 0.0;
        end
    end
end

% record the Tucker factors
for i = 1:N
    u = U{i};
    u = u(:,1:R(i));
    U{i} = u;
end
G = tmprod(A,U,1:N,'T');
T.core = G;
T.U = U;
rankss = R;
% relres = sqrt(1 - frob(G)^2/normA^2);

% print the result
% fprintf('Rank is %d\n',rankss);
% fprintf('Approximation error is %f\n',relres);
end
