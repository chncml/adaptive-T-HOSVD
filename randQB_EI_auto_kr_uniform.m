function [Q, B, mu] = randQB_EI_auto_kr_uniform(A, relerr, b, P)
    m  = size(A);n = length(m);
    mm = m(1);mn = prod(m(2 : n));
    Q = zeros(mm, 0);B = zeros(0, mn);
    E = norm(A(:))^2;A1 = reshape(A, mm, []);A = tensor(A);
    threshold = relerr;
    maxiter = ceil(min(mm, mn)/b);
    flag = false;Omg = cell(1, n);Omg{1} = [];
    for i = 1:maxiter
        for ind = 2 : n
            Omg{ind} = rand(m(ind), b) * 2 - 1;
        end
        Y = mttkrp(A, Omg, 1) - Q * (B * kr(Omg([2:n])));
        [Qi, ~] = qr(Y, 0);
        for j = 1:P
            [Qi, ~] = qr(A1'*Qi - B'*(Q'*Qi), 0);
            [Qi, ~] = qr(A1*Qi - Q*(B*Qi), 0);
        end
        [Qi, ~] = qr(Qi - Q * (Q' * Qi), 0);
        Bi = Qi' * A1 - Qi' * Q * B;
        Q = [Q, Qi];B = [B; Bi];
        temp = E- norm(Bi, 'fro')^2;
        if temp < threshold
            for j = 1:b
                E = E-norm(Bi(j,:))^2;
                if E< threshold
                    flag = true;
                    break;
                end
            end
        else
            E= temp;
        end
        if flag
            mu = (i - 1) * b + j;
            break;
        end
    end
    if ~flag
        mu = i * b;
    end
    Q = Q(:, 1 : mu);B = B(1 : mu, :);
end