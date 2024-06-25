function [F, TTr] = rttsms(A, k, varargin)
% Randomized Tucker decomposition of tensor A via single-mode sketching.
% Behnam Hashemi and Yuji Nakatsukasa, November 2023
% 
% Reference: 
% B. Hashemi and Y. Nakatsukasa, RTSMS: Randomized Tucker with single-mode
% sketching, submitted, 2023.
%
% Inputs
%     A: original tensor (d modes)
%     k: sketching parameter in least-squares. Typically 4
% order: processing order vector of modes (can be less than length d)
%
% The next input is expected to be a tolerance so that the input syntax is
%           [core_new, F] = rtsms(A, k, order, tol);
%  where tol is the rank determination tolerance.
% If, instead, a target rank is already available, then the expected input
% syntax is 
%           [core_new, F] = rtsms(A, k, order, r, 'rank');
%
% Outputs
%  core: core tensor (will be of size r)
%     F: cell array of factor matrices

sz = getsize(A);
core_old = A;
d = numel(sz);
F = cell(1,d);
TTr = ones(1,d);

len_inp = length(varargin);
if strcmpi(varargin{len_inp}, 'rank') % multilinear rank is specified.
    rankgiven = 1; 
    r1 = varargin{len_inp-1};
    tol = eps; % use default machine epsilon as tolerance in regularization
    r1tilde = round(1.5*r1);          % consistent with r + p where p = r/2
    TTr(2:end) = r1tilde;
%     r1tilde = min([r1tilde; sz]);     % r1tilde(i) should not be larger than sz(i)    
else
    rankgiven = 0;
    tol = varargin{end};              % tol is specified. We should estimate rank.
end

% Form core of size r1tilde(1) x r1tilde(2) x ... x r1tilde(d)
% and factors accordingly of size n(j) x r1tilde(j) 
for j = 1:d-1
    sz_comp = getsize(core_old); N = prod(sz_comp(2:end));
    rftCore = [];

    if rankgiven == 0
        r1tildenow = 10;
        rankfound = 0;
        iter = 0;
        while rankfound == 0
            iter = iter+1;
            % FIRST estimate the j-th multilinear rank!
            rforest = round(r1tildenow*1.1); % includes buffer space

            Omega = randn(rforest, TTr(j) * sz(j));
            % Our pagetmprod is faster than tmprod.
            core_new = tmprod(core_old, Omega, 1);
%             core_new = pagetmprod(core_old, Omega, mode);
            core_new_mode = tens2mat(core_new, 1); % fat mat
    
            dd = sign(randn(N, 1))';
            ix = randsample(N,min(N, k*(rforest)));
            extract = @(X) X(:,ix);   
            if isreal(core_new_mode)
                rft = @(u)dct(u.*dd,[],2,'type',4);  % keep the FFT'd matrix for possible later use
                % srft = @(u) extract(dct(u.*dd,[],2,'type',4)); 
            else
                rft = @(u) fft(dd.*u);
                %srft = @(u) extract(fft(dd.*u));         
            end
    
            if size(rftCore,1) > 20 % reuse previous RFT
                rftCoreadd = rft(core_new_mode(size(rftCore,1)+1:end,:));
                rftCore = [rftCore;rftCoreadd];
            else
                rftCore = rft(core_new_mode);
            end
    
            Snew = extract(rftCore); % This is analogous to XAY or XATheta in GN
            % We will use Snew only for finding leverage scores.
            [~,RR] = qr(Snew',0);  s = svd(RR);
    
            el = find(s/s(1) > tol,1,'last');
            if el < numel(s)/1.1 || rforest>= TTr(j)*sz(j)  % 1.1 respects rankest oversampling 
                rankfound = 1; 
                el = min(round(el*1.5), TTr(j)*sz(j));      % GN oversampling
                r1tildenow = el;
                if el <= rforest/1.1 % truncate
                    core_new_mode = core_new_mode(1:el,:); 
                    RR = RR(1:el,1:el);
                    ind = size(core_new);
                    size_trunc_core = [el  ind([j+1:end])];
                    trunc_indices = arrayfun(@(n) 1:n, size_trunc_core, 'UniformOutput', false);
                    core_new = core_new(trunc_indices{:});
                else % not enough for oversample, sketch extra
                    Omega_add = randn(el - rforest, TTr(j)*sz(j));
                    core_new_add = tmprod(core_old, Omega_add, 1); 

                    core_new_mode_add = tens2mat(core_new_add, 1); % fat mat   
                    core_new = cat(1,core_new,core_new_add);       % add new sketch
                    core_new_mode = [core_new_mode; core_new_mode_add];

                    if numel(core_new_mode_add)>0
                        rftCore_add = rft(core_new_mode_add);
                        Snew_add = extract(rftCore_add); % This is analogous to XAY or XATheta in GN
                        Snew = [Snew; Snew_add];
                    end
                    % We will use Snew only for finding leverage scores.
                    [~,RR] = qr(Snew',0);
                end
            else % Failed, increase rank estimate
                r1tildenow = min(round(r1tildenow*(1.7)),TTr(j)*sz(j));
            end
        end
    TTr(j+1) = r1tildenow;
    else % rank r1tilde is given as input        
         % Step 1: Simple construction of the new (temporary) core tensor,
         % smaller than core_old in one more dimension.
        r1tildenow = r1tilde(j);
        Omega = randn(r1tilde(j), TTr(j) * sz(j));
%         core_new = pagetmprod(core_old, Omega, mode); % Our pagetmprod is faster than tmprod.
        core_new = tmprod(core_old, Omega, 1);

        % Step 2: Start the process of finding the corresponding factor matrix. 
        % We use sketeched least-squares, regularization and iterative refinement.
        core_new_mode = tens2mat(core_new, 1);
    
        % When j = 1 and d = 3 we are now working with the old tensor without
        % unfolding it. If d >= 4, at the moment we simply unfold as case-by-case analysis is burdensome.

        dd = sign(randn(N, 1))';
        ix = randsample(N,min(N, k*(r1tilde(j))));
        extract = @(X) X(:,ix);
        if isreal(core_new_mode)
            %  srft = @(u) extract(dct(u.*dd,[],2,'type',4)); 
            srft = @(u) extract(fft(u.*dd,[],2));     % this was faster
        else
            srft = @(u) extract(fft(dd.*u)); 
        end
        Snew = srft(core_new_mode); % This is analogous to XAY or XATheta in GN

        % We will use Snew only for finding leverage scores.
        [~,RR] = qr(Snew',0);
    end
    
    levScores = (randn(5,size(RR,2))/(RR'))*core_new_mode;
    if j==1
        ix = leverage_score_sample_with_replacement(levScores', 4*k* r1tildenow);% twice for iterative refinement
    else
        ix = leverage_score_sample_with_replacement(levScores',3*k* r1tildenow);% twice for iterative refinement    
    end
    
    regtol = tol/5; 
    ix1 = unique(ix(1:round(numel(ix)/2)));     % For regularization
    ix2 = unique(ix(round(numel(ix)/2)+1:end)); % For iterative refinement
    regnow = normest(core_new_mode(:,ix1),0.1)*regtol;
    % core_new_mode(:,ix1) represents Omega_i A_{(i)} S_1 in the paper.
    
    % Tikhonov regularized least-squares
    [QQ,RR] = qr([core_new_mode(:,ix1)'; regnow*eye(size(core_new_mode,1))],0);   % best so far
    % Actual LS for computing Ftmp done below case-by-case

    % If d == 3, see the next instead! Those compute Ftmp and Bnow without core_old_mode.
%     if d >= 4
      core_old_mode = tens2mat(core_old, 1);
      Ftmp = (core_old_mode(:,ix1)* QQ(1:length(ix1),:))/RR';

        % This is where we do iterative refinement for d >= 4. 
      Bnow = core_old_mode(:,ix2) - Ftmp * core_new_mode(:,ix2);

%     elseif d == 3
%         core_old_ix1 = zeros(TTr(j)*sz(j), numel(ix1));
%         core_old_ix2 = zeros(TTr(j)*sz(j), numel(ix2));
% 
%         % Actual LS regularization here for d = 3
%         if mode == 3
%             % Create a matrix of indices using linear indexing and implicit expansion
%             indices1Matrix = ix1 + (0:sz_comp(1)*sz_comp(2):numel(core_old)-1);
%             core_old_ix1 = core_old(indices1Matrix);
% 
%             indices2Matrix = ix2 + (0:sz_comp(1)*sz_comp(2):numel(core_old)-1);
%             core_old_ix2 = core_old(indices2Matrix);
% 
%         elseif mode == 2
%             [iix1, jjx1] = ind2sub(sz_comp, ix1);
%             % Vectorized. Linear indices are non-obvious for modes other than 3 
%             % but we can work with subscipt indices iix1 and jjx1.
%             core_old_ix1 = cell2mat(arrayfun(@(i,j) core_old(i,:,j),iix1,jjx1, 'uniformoutput',0));
% 
%             [iix2, jjx2] = ind2sub(sz_comp, ix2);
%             core_old_ix2 = cell2mat(arrayfun(@(i,j) core_old(i,:,j),iix2,jjx2, 'uniformoutput',0));
% 
%         elseif mode == 1
%             [iix1, jjx1] = ind2sub(sz_comp, ix1);
%             core_old_ix1 = cell2mat(arrayfun(@(i,j) core_old(:,i,j),iix1,jjx1,'uniformoutput',0));
%             core_old_ix1 = reshape(core_old_ix1,[sz(mode), numel(ix1)])';
% 
%             [iix2, jjx2] = ind2sub(sz_comp, ix2);
%             core_old_ix2 = cell2mat(arrayfun(@(i,j) core_old(:,i,j),iix2,jjx2,'uniformoutput',0));
%             core_old_ix2 = reshape(core_old_ix2,[sz(mode), numel(ix2)])';
%         end
% 
%         Ftmp = (core_old_ix1' * QQ(1:length(ix1),:))/RR';
%         % This is the end of regularization for d = 3.
% 
%         Bnow = core_old_ix2' - Ftmp * core_new_mode(:,ix2);
%         % This is the end of iterative refinement for d = 3 
%         % without needing core_old_mode
%     end

    [QQ,RR] = qr([core_new_mode(:,ix2)'; regnow*eye(size(core_new_mode,1))],0);   
    % regnow is a pretty big regularization for this bit--makes sense as 
    % the RHS is possibly pretty small already.
    Fadd = (Bnow * QQ(1:length(ix2),:))/RR';
    TTr(j+1) = max(TTr(j+1),size(Ftmp + Fadd,2));
%     [TTr(j), sz(j), TTr(j+1)]
    
    F{j} = reshape(Ftmp + Fadd, [TTr(j), sz(j), TTr(j+1)]);
    if j<d-1
        core_old = reshape(core_new, [TTr(j+1) * sz(j+1), sz(j+2:d)]);  % Update the core tensor
    end
end
F{d} = reshape(core_new, [TTr(d), sz(d)]);
TTr = TTr(2:d);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%