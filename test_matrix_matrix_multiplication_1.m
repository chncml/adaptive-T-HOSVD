function test_matrix_matrix_multiplication_1
clc;clear;

dim = 800;
r = [20:20:200];
t11 = [];t12 = [];t13 = [];t14 = [];t15 = [];t16 = [];
A = randn(dim, dim, dim);
for k = 1 : length(r)
    k
    for sample = 1 : 10
        sample
        t = tic;
        c{1} = [];
        c{2} = randn(dim, r(k));
        c{3} = randn(dim, r(k));
        B1 = mttkrp(tensor(A), c, 1);
        t = toc(t);t11(k, sample) = t;

        t = tic;
        x = randn(dim, r(k));
        y = randn(dim, r(k));
        Amat = reshape(A, dim, []);
        z2 = kr(x, y);B2 = Amat * z2;
        t = toc(t);t12(k, sample) = t;

        t = tic;z3 = randn(dim^2, r(k));
        Amat = reshape(A, dim, []);
        B3 = Amat * z3;
        t = toc(t);t13(k, sample) = t;

        t = tic;
        c{1} = [];
        c{2} = rand(dim, r(k)) * 2 - 1;
        c{3} = rand(dim, r(k)) * 2 - 1;
        B4 = mttkrp(tensor(A), c, 1);
        t = toc(t);t14(k, sample) = t;

        t = tic;
        x = rand(dim, r(k)) * 2 - 1;
        y = rand(dim, r(k)) * 2 - 1;
        z5 = kr(x, y);
        Amat = reshape(A, dim, []);
        B5 = Amat * z5;
        t = toc(t);t15(k, sample) = t;

        t = tic;z6 = randi(2, [dim^2, r(k)]) * 2 - 3;
        Amat = reshape(A, dim, []);
        B6 = Amat * z6;
        t = toc(t);t16(k, sample) = t;
    end
end
dim1 = 200;
r = [10 : 10 : 100];
A = randn(dim1, dim1, dim1, dim1);
t21 = [];t22 = [];t23 = [];t24 = [];t25 = [];t26 = [];
for k = 1 : length(r)
    k
    for sample = 1 : 10
        sample
        t = tic;
        c{1} = [];
        c{2} = randn(dim1, r(k));
        c{3} = randn(dim1, r(k));
        c{4} = randn(dim1, r(k));
        B1 = mttkrp(tensor(A), c, 1);
        t = toc(t);t21(k, sample) = t;

        t = tic;
        x1 = randn(dim1, r(k));
        x2 = randn(dim1, r(k));
        x3 = randn(dim1, r(k));
        z2 = kr(x1, x2);z2 = kr(z2, x3);
        Amat = reshape(A, dim1, []);
        B2 = Amat * z2;
        t = toc(t);t22(k, sample) = t;

        t = tic;
        z3 = randn(dim1^3, r(k));
        Amat = reshape(A, dim1, []);
        B3 = Amat * z3;
        t = toc(t);t23(k, sample) = t;

        t = tic;
        c{1} = [];
        c{2} = rand(dim1, r(k)) * 2 - 1;
        c{3} = rand(dim1, r(k)) * 2 - 1;
        c{4} = rand(dim1, r(k)) * 2 - 1;
        B4 = mttkrp(tensor(A), c, 1);
        t = toc(t);t24(k, sample) = t;

        t = tic;
        x1 = rand(dim1, r(k)) * 2 - 1;
        x2 = rand(dim1, r(k)) * 2 - 1;
        x3 = rand(dim1, r(k)) * 2 - 1;
        z5 = kr(x1, x2);z5 = kr(z5, x3);
        Amat = reshape(A, dim1, []);
        B5 = Amat * z5;
        t = toc(t);t25(k, sample) = t;

        t = tic;
        z6 = rand(dim1^3, r(k)) * 2 - 1;
        Amat = reshape(A, dim1, []);
        B6 = Amat * z6;
        t = toc(t);t26(k, sample) = t;
    end
end
format short e
[sum(t11, 2)/10, sum(t12, 2)/10, sum(t13, 2)/10, sum(t14, 2)/10, sum(t15, 2)/10, sum(t16, 2)/10]
[sum(t21, 2)/10, sum(t22, 2)/10, sum(t23, 2)/10, sum(t24, 2)/10, sum(t25, 2)/10, sum(t26, 2)/10]
end