function W = cal_similarity_knn(X, k, is_normalized, is_symmetric)
    % Author: Hongyuan Zhang
    % X: d \times n
    % k: k-neighbors

    if ~exist('is_symmetric', 'var')
        is_symmetric = true;
    end
    if ~exist('is_normalized', 'var')
        is_normalized = false;
    end
    W = EuDist2(X', X', false);
    
    if exist('k','var')
        t = sort(W, 2, 'ascend');
        W_bool = W > t(:, k + 1);
        W(W_bool) = 0;
        W(~W_bool) = 1;
    end
    
    W = W - diag(diag(W));
    if is_normalized
        W = W ./ sum(W, 2);
    end
    if is_symmetric
        W = (W + W') / 2;
    end
end