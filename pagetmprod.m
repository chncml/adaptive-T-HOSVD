function S = pagetmprod(T,M,mode)
%PAGETMPROD Mode-n tensor-matrix product.
%
%   Behnam Hashemi and Yuji Nakatsukasa, 30 July 2023.
%   pagetmprod computes modal product of tensor T with matrix M using 
%   MATLAB's new built-in "tensorprod" function introduced in MATLAB R2022a.
%   It also uses MATLAB's "pagetranspose" function introduced in MATLAB R2020b.
%   The idea is to avoid explicitly reshaping and transposing arrays.
%
%   At the moment, this implementation works only for real tensors of order 3 and 4.
%        size(T, mode) should be equal to size(M, 2). 
%   The output is a tensor S such that 
%        mode-n unfolding of S equals M * mode-n unfolding of T, i.e.,
%        tens2mat(S, mode(n)) = U{n} * tens2mat(T, mode(n)).

sz_T = size(T); 
if size(T,3) == 1
    % Take care of case of n1 x n2 x 1 core tensor
    sz_T = [sz_T, 1];
end
d = numel(sz_T);

if d >= 5
    S = tmprod(T, M, mode);
end
if ~iscell(M) && size(M, 2) ~= sz_T(mode)
    error('size(T, mode) should be equal to size(M,2) = %i.',size(M,2));
end

% Check if 'tensorprod' and 'pagetranspose' are available 
if ~isMATLABReleaseOlderThan("R2022a")
    if numel(mode) == 1
        if mode == 1
            S = tensorprod(M, T, 2, mode);

        elseif mode == 2
            S = pagetranspose(tensorprod(M, T, 2, mode));

        elseif mode == 3 && d == 3
            % using pagemtimes is not obvious
            S = tensorprod(T, M, mode, 2);

        elseif mode == 3 && d == 4
            S = tensorprod(T, M, mode, 2);
            S = permute(S,[1 2 4 3]);

        elseif mode == 4 && d == 4
            S = tensorprod(T, M, mode, 2);
            % will need an extra permutation if T is 5-dimensional.
        end

     elseif numel(mode) > 1
         % M should be a cell array of factor matrices.
         % Call this code recursively. 
         S = pagetmprod(T, M{mode(1)}, mode(1));
         for j = 2:numel(mode)
             S = pagetmprod(S, M{mode(j)}, mode(j));
         end
    end

else
   warning('rtsms:tensorprod', ...
          ['MATLAB R2022a introduced tensorprod.\n' ...
           'Calling TensorLab''s tmprod instead.\n']);
         S = tmprod(T, M, mode);
end

end