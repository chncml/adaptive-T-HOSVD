function test_matrix_matrix_multiplication
dim = [100 : 50 : 800];
r = 100;
t11 = [];t12 = [];t13 = [];t14 = [];t15 = [];t16 = [];
for k = 1 : length(dim)
    k
    A = randn(dim(k), dim(k), dim(k));
    for sample = 1 : 10
        sample
        t = tic;
        c{1} = [];
        c{2} = randn(dim(k), r);
        c{3} = randn(dim(k), r);
        B1 = mttkrp(tensor(A), c, 1);
        t = toc(t);t11(k, sample) = t;

        t = tic;
        x = randn(dim(k), r);
        y = randn(dim(k), r);
        Amat = reshape(A, dim(k), []);
        z2 = kr(x, y);B2 = Amat * z2;
        t = toc(t);t12(k, sample) = t;

        t = tic;z3 = randn(dim(k)^2, r);
        Amat = reshape(A, dim(k), []);
        B3 = Amat * z3;
        t = toc(t);t13(k, sample) = t;

        t = tic;
        c{1} = [];
        c{2} = rand(dim(k), r) * 2 - 1;
        c{3} = rand(dim(k), r) * 2 - 1;
        B4 = mttkrp(tensor(A), c, 1);
        t = toc(t);t14(k, sample) = t;

        t = tic;
        x = rand(dim(k), r) * 2 - 1;
        y = rand(dim(k), r) * 2 - 1;
        z5 = kr(x, y);
        Amat = reshape(A, dim(k), []);
        B5 = Amat * z5;
        t = toc(t);t15(k, sample) = t;

        t = tic;z6 = randi(2, [dim(k)^2, r]) * 2 - 3;
        Amat = reshape(A, dim(k), []);
        B6 = Amat * z6;
        t = toc(t);t16(k, sample) = t;
    end
end
format short e
[sum(t11, 2)/10, sum(t12, 2)/10, sum(t13, 2)/10, sum(t14, 2)/10, sum(t15, 2)/10, sum(t16, 2)/10]
dim1 = [100 : 10 : 200];
r = 50;
t21 = [];t22 = [];t23 = [];t24 = [];t25 = [];t26 = [];
for k = 1 : length(dim1)
    k
    A = randn(dim1(k), dim1(k), dim1(k), dim1(k));
    for sample = 1 : 10
        sample
        t = tic;
        c{1} = [];
        c{2} = randn(dim1(k), r);
        c{3} = randn(dim1(k), r);
        c{4} = randn(dim1(k), r);
        B1 = mttkrp(tensor(A), c, 1);
        t = toc(t);t21(k, sample) = t;

        t = tic;
        x1 = randn(dim1(k), r);
        x2 = randn(dim1(k), r);
        x3 = randn(dim1(k), r);
        z2 = kr(x1, x2);z2 = kr(z2, x3);
        Amat = reshape(A, dim1(k), []);
        B2 = Amat * z2;
        t = toc(t);t22(k, sample) = t;

        t = tic;
        z3 = randn(dim1(k)^3, r);
        Amat = reshape(A, dim1(k), []);
        B3 = Amat * z3;
        t = toc(t);t23(k, sample) = t;

        t = tic;
        c{1} = [];
        c{2} = rand(dim1(k), r) * 2 - 1;
        c{3} = rand(dim1(k), r) * 2 - 1;
        c{4} = rand(dim1(k), r) * 2 - 1;
        B4 = mttkrp(tensor(A), c, 1);
        t = toc(t);t24(k, sample) = t;

        t = tic;
        x1 = rand(dim1(k), r) * 2 - 1;
        x2 = rand(dim1(k), r) * 2 - 1;
        x3 = rand(dim1(k), r) * 2 - 1;
        z5 = kr(x1, x2);z5 = kr(z5, x3);
        Amat = reshape(A, dim1(k), []);
        B5 = Amat * z5;
        t = toc(t);t25(k, sample) = t;

        t = tic;
        z6 = rand(dim1(k)^3, r) * 2 - 1;
        Amat = reshape(A, dim1(k), []);
        B6 = Amat * z6;
        t = toc(t);t26(k, sample) = t;
    end
end
% dim = [100 : 50 : 800];
% r = 100;
% t31 = [];t32 = [];t33 = [];t34 = [];
% for k = 1 : length(dim)
%     k
%     A1 = randn(dim(k), dim(k));A2 = randn(dim(k), dim(k));
%     Amat = kr(A1, A2);Amat = Amat';
%     for sample = 1 : 10
%         sample
%         t = tic;
%         x = randn(dim(k), r);y = randn(dim(k), r);
%         B1 = (A1'*x).*(A2'*y);
%         t = toc(t);t31(k, sample) = t;
% 
%         t = tic;z2 = randn(dim(k)^2, r);
%         B2 = Amat * z2;
%         t = toc(t);t32(k, sample) = t;
% 
%         t = tic;
%         x = rand(dim(k), r) * 2 - 1;
%         y = rand(dim(k), r) * 2 - 1;
%         B3 = (A1'*x).*(A2'*y);
%         t = toc(t);t33(k, sample) = t;
% 
%         t = tic;z4 = rand(dim(k)^2, r) * 2 - 1;
%         B4 = Amat * z4;
%         t = toc(t);t34(k, sample) = t;
%     end
% end
% dim1 = [30 : 30 : 300];
% t41 = [];t42 = [];t43 = [];t44 = [];
% for k = 1 : length(dim1)
%     k
%     A1 = randn(dim1(k));A2 = randn(dim1(k));A3 = randn(dim1(k));
%     Amat = kr(A1, kr(A2, A3));Amat = Amat';
%     for sample = 1 : 10
%         sample
%         t = tic;
%         x1 = randn(dim1(k), r);x2 = randn(dim1(k), r);x3 = randn(dim1(k), r);
%         B1 = (A1' * x1).*(A2' * x2).*(A3' * x3);
%         t = toc(t);t41(k, sample) = t;
% 
%         t = tic;z2 = randn(dim1(k)^3, r);
%         B2 = Amat * z2;
%         t = toc(t);t42(k, sample) = t;
% 
%         t = tic;
%         x1 = rand(dim1(k), r) * 2 - 1;
%         x2 = rand(dim1(k), r) * 2 - 1;
%         x3 = rand(dim1(k), r) * 2 - 1;
%         B3 = (A1' * x1).*(A2' * x2).*(A3' * x3);
%         t = toc(t);t43(k, sample) = t;
% 
%         t = tic;z4 = rand(dim1(k)^3, r) * 2 - 1;
%         B4 = Amat * z4;
%         t = toc(t);t44(k, sample) = t;
%     end
% end
format short e
[sum(t21, 2)/10, sum(t22, 2)/10, sum(t23, 2)/10, sum(t24, 2)/10, sum(t25, 2)/10, sum(t26, 2)/10]
end