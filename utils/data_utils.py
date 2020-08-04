import numpy as np

def transformation(series, code, transform=True, scale=1):
    transformed_series = np.zeros(series.shape[0])

    if transform:
        if code == 1:
            # none
            transformed_series = series
        elif code == 2:
            # first-difference
            transformed_series[1:] = series[1:] - series[:-1]
        elif code == 3:
            # second-difference
            transformed_series = series[2:] - series[:-2]
        elif code == 4:
            # log
            transformed_series = np.log(series)*scale
        elif code == 5:
            # first-difference log
            transformed_series[1:] = (np.log(series[1:]) -
                                      np.log(series[:-1]))*scale
        elif code == 6:
            # second-difference log
            transformed_series[2:] = (np.log(series[2:]) - 2*np.log(series[1:-1])
                                     + np.log(series[:-2]))*scale

        return transformed_series
    else:
        return series


def create_data(series, train_indices, T, M, p, k):
    # Create data
    lagged_y = np.ones((T, M * p + 1))
    lagged_series = []
    y_series = []

    for s in series:
        lagged_series.append(s[:train_indices])

    position_counter = 1  # Constant is added in front

    for m in range(M):
        y_m = lagged_series[m]
        for i in range(1, p + 1):
            lagged_y[:, position_counter] = y_m[(p - i):-i]
            position_counter += 1

    # Create lagged dependent matrix
    X = np.zeros((T, M, k))
    stacked_X = np.zeros((M, T, k))

    for m in range(M):
        total_lags = M * p + 1
        stacked_X[m, :, m * (total_lags):(m + 1) * total_lags] = lagged_y

    stacked_list = list()

    for m in range(M):
        stacked_list.append(stacked_X[m])

    for t in range(T):
        X[t] = np.squeeze(np.dstack(tuple(stacked_list)))[t].T

    for s in series:
        y_series.append(s[p:train_indices])

    y = np.array(y_series)
    y = y.T

    return X, y

def create_dgp_data(T, M, p):
    # Create data
    lagged_y = np.zeros((T, M * p))
    k = M(M*p+1)

    locations = np.random.randint(0, 10, size=M)
    position_counter = 0

    for m in range(M):
        y_m = np.random.normal(loc=locations[m], scale=2, size=T + p)
        for i in range(1, p + 1):
            lagged_y[:, position_counter] = y_m[(p - i):-i]
            position_counter += 1

    # Create lagged dependent matrix
    X = np.zeros((T, M, k))
    stacked_X = np.zeros((M, T, k))

    for m in range(M):
        total_lags = M * p
        stacked_X[m, :, m * (total_lags):(m + 1) * total_lags] = lagged_y

    stacked_list = list()

    for m in range(M):
        stacked_list.append(stacked_X[m])

    for t in range(T):
        X[t] = np.squeeze(np.dstack(tuple(stacked_list)))[t].T

    # Create betas
    ub = 5
    lb = 0
    difference = 0.5
    scale = 0.010
    sign = -1

    beta = np.zeros((T, k))

    for i in range(k): 
        bound = np.random.randint(lb, ub)

        if sign == 1:
            sign = -1
        else:
            sign = 1

        beta[:, i] = np.linspace(bound, bound + sign * difference, T) + np.random.normal(scale=scale, size=T)

        # beta[:,27:32] = 0
        beta[50:125, 2] = 0

    # Construct dependent
    y = np.zeros((M, T))

    for i in range(T):
        y[:, i] = X[i] @ beta[i] + np.random.normal(
            size=M).T  # np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag([0.10,5]))

    # Transpose for KF
    y = y.T

    return X, y

def standardize(series, train):

    standardized_series = []

    for individual_series in series:
        standardized = (individual_series - individual_series[:train].mean())/individual_series[:train].std()

        standardized_series.append(standardized)

    return standardized_series