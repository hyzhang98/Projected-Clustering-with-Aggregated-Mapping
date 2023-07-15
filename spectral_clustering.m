function [y_pred, F, L, D] = spectral_clustering(W, cluster_count, ratio_cut)
    % Input:
    % W: n * n, graph matrix (or namely, similarity matrix), which is used to build Laplacian
    % cluster_count: no. of clusters
    % ratio_cut: bool (false as default), use ratio_cut if it is true, or use normalized cut otherwise. 
    % 
    % Output:
    % y_pred: clustering assignment
    % Authored by Hongyuan Zhang
    if ~exist('ratio_cut', 'var')
        ratio_cut = false;
    end
    D = max(sum(W), 10^-6);
    D = diag(D);
    [n, ~] = size(W);
    I = eye(n);
    if ratio_cut
        L = D - W;
    else
        % sqrtD = diag(D^-0.5);
        % L = I - sqrtD * W * sqrtD;
        sqrtD = diag(D).^-0.5;
        L = I - sqrtD' .* W .* sqrtD;
    end
    L = sparse(L);
    L = max(L, L');
    % [F, S] = eig(L); % F: n * k
    % [~, idx] = sort(diag(S));
    % F = F(:, idx);
    % F = F(:, 1: cluster_count);
    [F, ~] = eigs(L, cluster_count, 'smallestreal');
    row_norms = sqrt(sum(F.^2, 2));
    idx = find(row_norms ~= 0);
    F_for_pred = F;
    F_for_pred(idx, :) = F_for_pred(idx, :) ./ row_norms(idx); 
    F_for_pred = F_for_pred ./ sqrt(sum(F_for_pred.^2, 2));
    y_pred = kmeans(F_for_pred, cluster_count, 'Replicates', 20);
end