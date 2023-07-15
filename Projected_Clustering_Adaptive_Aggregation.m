function [y_pred, Z, S, obj1, obj2] = Projected_Clustering_Adaptive_Aggregation(X, n_clusters, dim_feature, k_0, var_beta, is_increase, is_fixed, is_sc, option)
    % Input:
    %   X: d *n 
    %   n_clustering: amount of clusters
    %   dim_features: dimension of projection space
    %   k_0: initial sparsity
    %   var_beta: balance coefficient for Local Structure, referring to beta in the paper
    %   is_sc: true - use sc; false - use kmeans
    %   is_increase: whether increase sparsity
    %   option.flag: Choose how to construct the graph 
    %       if flag == 0 then using CAN (by default)
    %       elseif flag == 1 then using Gaussian
    %       else using knn
    %   option.iter: The iteration number to update graph. The default nubmer is 5. 
    %   option.k_max: The upper-bound of neighbors. The default bound is n/(2*n_clusters)
    % Output: 
    %   y_pred: prediction
    % 
    % Author: Hongyuan Zhang
    [~, n] = size(X);
    if ~exist('is_sc', 'var')
        is_sc = true;
    end
    if ~exist('is_fixed', 'var')
        is_fixed = false;
    end
    if ~exist('is_increase', 'var')
        is_increase = true;
    end
    if ~exist('option', 'var')
        option = struct();
    end
    if ~isfield(option, 'flag')
        option.flag = 0;
    end
    if ~isfield(option, 'iter')
        option.iter = 5;
    end
    if ~isfield(option, 'k_max')
        option.k_max = floor(n / n_clusters);
    end
    [~, n] = size(X);
    k = k_0;
    if option.flag == 0
        S = cal_similarity_can(X, k);
    elseif option.flag == 1
        S = cal_similarity_Gaussian(X, option.gaussian_gamma, k, true, false);
    else
        S = cal_similarity_knn(X, k, true, false);
    end
    S = sparse(S);
    S = S';
    Z = X;
    inc_step = 0;
    obj1 = zeros(option.iter, 1);
    obj2 = zeros(option.iter, 1);
    if is_increase
        inc_step = floor((option.k_max - k) / (option.iter));
    end
    for i = 1: option.k_max
        sym_S = (S + S') / 2;
        D = diag(sum(sym_S));
        L = D - sym_S; 
        Ab = double(S == 0) / (n - k);

        Qb = eye(n) - Ab;

        Q = eye(n) - S; 
        
        Q1 = X*Q;
        Q1 = Q1 * Q1';
        Q2 = X*Qb;
        Q2 = Q2 * Q2';
        Q = Q1 - Q2 + var_beta * X*L*X';
        Q = max(Q, Q');
        % It seems that eigs is too slow and senstive in some cases. 
        % We recommend to use eigs on large datasets. 
        % [W, ~] = eigs(Q, dim_feature, 'smallestreal');
        [eig_matrix, eig_values] = eig(Q);
        [~, idx] = sort(diag(eig_values));
        W = eig_matrix(:, idx);
        W = W(:, 1:dim_feature);

        k = k + inc_step; 
        Z = W' * X;

        if option.flag == 0
            S = cal_similarity_can(Z, k);
        elseif option.flag == 1
            S = cal_similarity_Gaussian(Z, option.gaussian_gamma, k, true, false);
        else
            S = cal_similarity_knn(Z, k, true, false);
        end
        S = S';
        if is_fixed
            break;
        end
    end
    if is_sc
        y_pred = spectral_clustering((S+S')/2, n_clusters);
    else
        y_pred = kmeans(Z', n_clusters, 'MaxIter', 100, 'Replicates', 20);
    end

end