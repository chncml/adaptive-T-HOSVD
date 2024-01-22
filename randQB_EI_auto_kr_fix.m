function [Q, B, k] = randQB_EI_auto_kr_fix(A, relerr, b, P)
% [Q, B, k] = randQB_EI_auto(A, relerr, b, P)
% The fixed-precision randQB_EI algorithm.
% It produces QB factorization of A, whose approximation error fulfills
%     ||A-QB||_F <= ||A||_F* relerr.
% b is block size, P is power parameter.
% Output k is the rank.
    m  = size(A);n = length(m);
    mm = m(1);mn = prod(m(2 : n));Q = zeros(mm, 0);
    B = zeros(0, mn);
    A = reshape(A, m(1), []);
    E = norm(A, 'fro')^2;
    threshold = relerr;
    maxiter = ceil(min(mm, mn)/b);
    flag = false;
    Omg = ones(1, b);
    for ind = 2 : n
        Omg = kr(Omg, randn(m(ind), b));
    end
    Ag = A * Omg;
    for i = 1:maxiter,
        
        Y = Ag - (Q * (B * Omg));
        [Qi, ~] = qr(Y, 0);
        
        for j = 1:P
            [Qi, ~] = qr(A'*Qi - B'*(Q'*Qi), 0);  % can skip orthonormalization for small b. 
            [Qi, ~] = qr(A*Qi - Q*(B*Qi), 0);
        end
        
        [Qi, ~] = qr(Qi - Q * (Q' * Qi), 0);
        
        Bi = Qi' * A - Qi' * Q * B;
        
        Q = [Q, Qi];
        B = [B; Bi];
        
        temp = E- norm(Bi, 'fro')^2;
        
        if temp< threshold,   % precise rank determination 
            for j=1:b,
                E= E-norm(Bi(j,:))^2;
                if E< threshold,
                    flag= true;
                    break;
                end
            end
        else
            E= temp;
        end
        if flag,
            k= (i - 1) * b + j;
            break;
        end
    end
    if ~flag,
        k= i * b;
%         fprintf('E = %f. Fail to converge within maxiter!\n\n', sqrt(E/E0));
    end
    Q = Q(:, 1 : k);B = B(1 : k, :);
end