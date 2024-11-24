function test_generate_random_matrices

dim = [50 : 50 : 800];
r = 100;
t11 = [];t12 = [];t13 = [];t14 = [];
for k = 1 : length(dim)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim(k), r);y = randn(dim(k), r);
        z1 = kr(x, y);t = toc(t);t11(k, sample) = t;

        t = tic;z2 = randn(dim(k)^2, r);
        t = toc(t);t12(k, sample) = t;

        t = tic;
        x = rand(dim(k), r) * 2 - 1;
        y = rand(dim(k), r) * 2 - 1;
        z3 = kr(x, y);t = toc(t);t13(k, sample) = t;
        
        t = tic;z4 = rand(dim(k)^2, r) * 2 - 1;
        t = toc(t);t14(k, sample) = t;
    end
end

dim1 = [30 : 30 : 300];
t21 = [];t22 = [];t23 = [];t24 = [];
for k = 1 : length(dim1)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1(k), r);x2 = randn(dim1(k), r);x3 = randn(dim1(k), r);
        z1 = kr(x1, x2);z1 = kr(z1, x3);
        t = toc(t);t21(k, sample) = t;

        t = tic;z2 = randn(dim1(k)^3, r);
        t = toc(t);t22(k, sample) = t;

        t = tic;
        x1 = rand(dim1(k), r) * 2 - 1;
        x2 = rand(dim1(k), r) * 2 - 1;
        x3 = rand(dim1(k), r) * 2 - 1;
        z3 = kr(x1, x2);z3 = kr(z3, x3);
        t = toc(t);t23(k, sample) = t;

        t = tic;z4 = rand(dim1(k)^3, r) * 2 - 1;
        t = toc(t);t24(k, sample) = t;
    end
end

dim = 800;r1 = [20 : 20 : 200];
t31 = [];t32 = [];t33 = [];t34 = [];
for k = 1 : length(r1)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x = randn(dim, r1(k));y = randn(dim, r1(k));
        z1 = kr(x, y);t = toc(t);t31(k, sample) = t;

        t = tic;z2 = randn(dim^2, r1(k));
        t = toc(t);t32(k, sample) = t;

        t = tic;
        x = rand(dim, r1(k)) * 2 - 1;
        y = rand(dim, r1(k)) * 2 - 1;
        z3 = kr(x, y);t = toc(t);t33(k, sample) = t;

        t = tic;z4 = rand(dim^2, r1(k)) * 2 - 1;
        t = toc(t);t34(k, sample) = t;
    end
end
dim1 = 300;r2 = [10 : 10 : 100];
t41 = [];t42 = [];t43 = [];t44 = [];
for k = 1 : length(r2)
    k
    for sample = 1 : 10
        sample
        t = tic;
        x1 = randn(dim1, r2(k));
        x2 = randn(dim1, r2(k));
        x3 = randn(dim1, r2(k));
        z1 = kr(x1, x2);z1 = kr(z1, x3);
        t = toc(t);t41(k, sample) = t;

        t = tic;z2 = randn(dim1^3, r2(k));
        t = toc(t);t42(k, sample) = t;

        t = tic;
        x1 = rand(dim1, r2(k)) * 2 - 1;
        x2 = rand(dim1, r2(k)) * 2 - 1;
        x3 = rand(dim1, r2(k)) * 2 - 1;
        z3 = kr(x1, x2);z3 = kr(z3, x3);
        t = toc(t);t43(k, sample) = t;

        t = tic;z4 = rand(dim1^3, r2(k)) * 2 - 1;
        t = toc(t);t44(k, sample) = t;
    end
end
format short e
[sum(t11, 2)/10, sum(t12, 2)/10, sum(t13, 2)/10, sum(t14, 2)/10]
[sum(t21, 2)/10, sum(t22, 2)/10, sum(t23, 2)/10, sum(t24, 2)/10]
[sum(t31, 2)/10, sum(t32, 2)/10, sum(t33, 2)/10, sum(t34, 2)/10]
[sum(t41, 2)/10, sum(t42, 2)/10, sum(t43, 2)/10, sum(t44, 2)/10]
end