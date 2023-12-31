function S = cal_similarity_can(X, k)
    % solve problem: \sum_{i, j} w_{ij} ||x_i - x_j||_2^2 + lam * ||W||_F^2
    % X: d * n (d, n represents the dimension and no. of samples,
    %           respectively)
    % k: sparsity (no. of neighbors)
    % Output:
    % S: n * n, similarity matrix
    % Authored by Hongyuan Zhang
    
    D = EuDist2(X', X', false);
    n = size(D, 1);
    sorted_D = sort(D, 2);
    k_D = sorted_D(:, k+1) + 10^-6;
    top_k_sum = sum(sorted_D(:, 1:k), 2);
    S = max(repmat(k_D, 1, n) - D, 0);
    S = S ./ repmat(k * k_D - top_k_sum, 1, n);
end