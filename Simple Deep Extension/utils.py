import torch
from data_loader import load_data

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0, descending=False):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    if size > 10000:
        d = distances[:5000, :]
        sort_d1, _ = d.sort(dim=1, descending=descending)
        torch.cuda.empty_cache()
        d = distances[5000:, :]
        sort_d2, _ = d.sort(dim=1, descending=descending)
        d = None
        torch.cuda.empty_cache()
        sorted_distances = torch.cat((sort_d1, sort_d2))
    else:
        sorted_distances, _ = distances.sort(dim=1, descending=descending)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links is not 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    device = X.device
    raw_weights = raw_weights.to(device)
    weights = weights.to(device)
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t()*degree


def noise(weights, ratio=0.1):
    sampling = torch.rand(weights.shape).cuda() + torch.eye(weights.shape[0]).cuda()
    sampling = (sampling > ratio).type(torch.IntTensor).cuda()
    return weights * sampling
