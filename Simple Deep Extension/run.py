from model import AdaGAE
import torch
import data_loader as loader
import warnings
import numpy as np
import scipy.io
warnings.filterwarnings('ignore')
import os

for dataset in [ loader.FASHION]:
    [data, labels] = loader.load_data(dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X = torch.Tensor(data).to(device)
    input_dim = data.shape[1]
    layers = None
    if dataset is loader.USPS:
        layers = [input_dim, 128, 64]
    else:
        layers = [input_dim, 256, 64]
    accs = []
    nmis = []
    inc = 0
    lam = 0.01
    for lam in [0.01] :#np.power(10.0, np.array(range(-3,3))):
        for neighbors in [50]:
            print('-----lambda={}, neighbors={}, inc={}'.format(lam, neighbors, inc))
            gae = AdaGAE(X, labels, layers=layers, num_neighbors=neighbors, lam=lam, max_iter=100, max_epoch=10,
                        update=True, learning_rate=10**-4, inc_neighbors=inc, device=device).to(device)
            acc, nmi = gae.run()
            
            accs.append(acc)
            nmis.append(nmi)

    scipy.io.savemat(f'{dataset}.mat', mdict={'acc': accs, 'nmi': nmis,})
    print(accs)
    print(nmis)