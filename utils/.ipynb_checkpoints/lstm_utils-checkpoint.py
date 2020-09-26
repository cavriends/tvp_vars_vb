import torch
import numpy as np

def create_sequences_lstm(series, lag):

    series = np.array(series)
    T = series.shape[1]
    M = series.shape[0]

    X = torch.Tensor(np.zeros((T, lag, M)))
    y = torch.Tensor(np.zeros((T, M)))

    sequenced_data = []

    for t in range(lag, T):

        X[t,:,:] = torch.Tensor(series[:,(t-lag):t]).T
        y[t,:] = torch.Tensor(series[:,t])

        sequenced_data.append((X[t,:,:], y[t,:]))

    return X, y, sequenced_data