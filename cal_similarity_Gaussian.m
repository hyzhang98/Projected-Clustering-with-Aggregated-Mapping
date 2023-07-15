function W = cal_similarity_Gaussian(X, gamma, k, is_normalized, is_symmetric)
    % Author: Hongyuan Zhang
    % X: d \times n
    % gamma: width parameter
    % k: k-neighbors

    if ~exist('is_symmetric', 'var')
        is_symmetric = true;
    end
    if ~exist('is_normalized', 'var')
        is_normalized = false;
    end
    W = EuDist2(X', X', false);
    
    % W = W / mean(W, 'all');
    W = exp(-W/gamma);
    
    if is_normalized
        W = W +  + 10^-6;
    end
    
    if exist('k','var')
        t = sort(W, 2, 'descend');
        W(W < t(:, k + 1)) = 0;
    end
    
    W = W - diag(diag(W));
    if is_normalized
        W = W ./ sum(W, 2);
    end
    if is_symmetric
        W = (W + W') / 2;
    end
end