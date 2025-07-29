from sklearn.model_selection import StratifiedKFold
import torch
import random
import numpy as np



def setup_seed(seed):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def data_split(X, Y, seed, K):
    # get split idx
    train_splits = []
    test_splits = []
    val_splits = []

    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)   #n_splits: The number of samples that cannot be exceeded in each category
    for train_val_idx, test_idx in kf.split(X=X, y=Y):                  #X:Feature data, Y:Label data
        kf_val = StratifiedKFold(n_splits=K-1, shuffle=True, random_state=seed)
        x = X[train_val_idx]
        y = Y[train_val_idx]
        for train_idx, val_idx in kf_val.split(X=x, y=y):
            test_splits.append(X[test_idx].tolist())
            train_splits.append(x[train_idx].tolist())
            val_splits.append(x[val_idx].tolist())
    for i, train_idx in enumerate(train_splits):
        assert set(train_idx + test_splits[i] + val_splits[i]) == set(X.tolist())

    # return train_splits[0::K - 1], [val_splits[0]], [test_splits[0]]
    return train_splits[0::K-1], val_splits[0::K-1], test_splits[0::K-1]

