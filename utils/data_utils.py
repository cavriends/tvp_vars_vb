import numpy as np
from scipy.stats import norm, multivariate_normal

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


def generate_dgp_tvp_var(M, T, p, diagonal_coefficient, cross_coefficient, sigma_states, sigma_observation, covariance,
                     binomial_prob):
    y = np.zeros((M, T))
    A_1_vec = np.zeros((M * (M * p), T))
    selection_mask = np.random.binomial(1, binomial_prob, M * (M * p)) == 1

    for t in range(T):

        if t == 0:
            A_1_vec[selection_mask, t] = cross_coefficient
            np.fill_diagonal(A_1_vec[:, t].reshape(M, (M * p)), np.repeat(diagonal_coefficient, M))
            y[:, t] = np.ones(M)
        else:
            A_1_vec[:, t] = A_1_vec[:, t - 1] + multivariate_normal.rvs(mean=np.zeros(M ** 2),
                                                                        cov=np.diag(np.ones(M ** 2) * sigma_states))

            ## Eigen values check
            eigen_values = np.linalg.eig(A_1_vec[:, t].reshape(M, M))[0]
            stationary = any(eigen_values < 1)
            if stationary == False:
                print(f'Failed eigen values requirement < 1 (explosive process)')
                print(f'Iteration: {t}')
                break

            Z = np.zeros((M, M ** 2))
            for m in range(M):
                Z[m, m * M:(m * M + M)] = y[:, t - 1]
            y[:, t] = Z @ A_1_vec[:, t] + multivariate_normal.rvs(mean=np.zeros(M), cov=(
                        np.diag(np.ones(M)) * sigma_observation + np.tril(np.repeat(covariance, M), -1)))

    return y, A_1_vec


def generate_dgp_tvp_var_heteroskedastic(M, T, p, diagonal_coefficient, cross_coefficient, sigma_states, sigma_observation, covariance,
                     binomial_prob, sigma_obs_cov, sigma_states_cov):
    y = np.zeros((M, T))
    A_1_vec = np.zeros((M * (M * p), T))
    selection_mask = np.random.binomial(1, binomial_prob, M * (M * p)) == 1
    selection_mask[np.diag_indices(M)[0]] = True

    sigma_states_vec = np.ones((T,M**2))*sigma_states
    sigma_observation_vec = np.ones((T,M))*sigma_observation
    sigma_observation_cross_vec = np.ones((T,((M**2)-M)//2))*sigma_observation

    for t in range(T):
        triangular_zeros = np.zeros((M,M))
        if t == 0:
            A_1_vec[selection_mask, t] = cross_coefficient
            np.fill_diagonal(A_1_vec[:, t].reshape(M, (M * p)), np.repeat(diagonal_coefficient, M))
            y[:, t] = np.ones(M)
        else:
            sigma_states_vec[t] = np.abs(sigma_states_vec[t-1] + multivariate_normal.rvs(mean=np.zeros(M**2),cov=sigma_states_cov*np.eye(M**2)))
            sigma_observation_vec[t] = np.abs(sigma_observation_vec[t-1] + multivariate_normal.rvs(mean=np.zeros(M),cov=sigma_obs_cov*np.eye(M)))
            sigma_observation_cross_vec[t] = sigma_observation_cross_vec[t-1] + multivariate_normal.rvs(mean=np.zeros(((M**2)-M)//2), cov=sigma_obs_cov*np.eye(((M**2)-M)//2))

            sigma_states_vec[t,~selection_mask] = 0
            A_1_vec[:, t] = A_1_vec[:, t - 1] + multivariate_normal.rvs(mean=np.zeros(M ** 2),
                                                                        cov=np.diag(sigma_states_vec[t]))

            ## Eigen values check
            eigen_values = np.linalg.eig(A_1_vec[:, t].reshape(M, M))[0]
            stationary = any(eigen_values < 1)
            if stationary == False:
                print(f'Failed eigen values requirement < 1 (explosive process)')
                print(f'Iteration: {t}')
                break

            Z = np.zeros((M, M ** 2))
            for m in range(M):
                Z[m, m * M:(m * M + M)] = y[:, t - 1]

            triangular_zeros[np.tril_indices(M,-1)] = sigma_observation_cross_vec[t]
            triangular_zeros[np.triu_indices(M,1)]  = sigma_observation_cross_vec[t]
            y[:, t] = Z @ A_1_vec[:, t] + multivariate_normal.rvs(mean=np.zeros(M), cov=(
                        np.diag(sigma_observation_vec[t]) + triangular_zeros))

    return y, A_1_vec, sigma_states_vec, sigma_observation_vec, sigma_observation_cross_vec

def generate_matrices(T, M, p, y):

    series = y

    lagged_T = T - p
    lagged_series = []
    y_series = []

    lagged_y = np.ones((lagged_T, M * p))
    k = M*(M*p)
    position_counter = 0
    total_lags = M * p

    for m in range(M):
        y_m = series[m]
        for i in range(1, p + 1):

            lagged_y[:, position_counter] = y_m[(p - i):-i]
            position_counter += 1

    # Create lagged dependent matrix
    X = np.zeros((lagged_T, M, k))
    stacked_X = np.zeros((M, lagged_T, k))

    for m in range(M):
        stacked_X[m, :, m * (total_lags):(m + 1) * total_lags] = lagged_y

    stacked_list = list()

    for m in range(M):
        stacked_list.append(stacked_X[m])

    for t in range(lagged_T):
        X[t] = np.squeeze(np.dstack(tuple(stacked_list)))[t].T

    for s in series:
        y_series.append(s[p:])

    y_own = np.array(y_series)
    X_own = X

    return y_own, X_own


def generate_contemp_matrices(T, M, p, y):
    # Contemperous values added

    series = y

    lagged_T = T - p
    lagged_series = []
    y_series = []

    lagged_y = np.ones((lagged_T, M * p))
    k = M * (M * p) + M * (M - 1)
    variable_list = np.arange(M)
    position_counter = 0
    total_lags = M * p + (M - 1)

    for m in range(M):
        y_m = series[m]
        for i in range(1, p + 1):
            lagged_y[:, position_counter] = y_m[(p - i):-i]
            position_counter += 1

    # Create lagged dependent matrix
    X = np.zeros((lagged_T, M, k))
    stacked_X = np.zeros((M, lagged_T, k))

    for m in range(M):
        contemp_y = np.zeros((T - 1, M - 1))

        if m != 0:
            contemp_y[:, :m] = series[:m][:, 1:].T

        stacked_X[m, :, m * (total_lags):(m + 1) * total_lags] = np.hstack((lagged_y, contemp_y))

    stacked_list = list()

    for m in range(M):
        stacked_list.append(stacked_X[m])

    for t in range(lagged_T):
        X[t] = np.squeeze(np.dstack(tuple(stacked_list)))[t].T

    for s in series:
        y_series.append(s[p:])

    y_own = np.array(y_series)
    X_own = X

    return y_own, X_own

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
    k = M*(M*p+1)

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

    return X, y, beta

def standardize(series, train):

    standardized_series = []

    for individual_series in series:
        standardized = (individual_series - individual_series[:train].mean())/individual_series[:train].std()

        standardized_series.append(standardized)

    return standardized_series