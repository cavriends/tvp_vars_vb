def calculate_predictions(total_h, current_X, mt1t):

    prev_X = current_X

    prev_pred = np.zeros((M,total_h))

    for h in range(total_h):

        prev_pred[:,h] = prev_X@mt1t[:,-1]
        vec_X = prev_X[0,:(M*p+1)]

        empty_X = np.zeros((M*p+1))
        empty_X[0] = 1

        for m in range(M):
            empty_X[(m*p+2):((m+1)*p+1)] = vec_X[(m*p+1):((m+1)*p)]

        vec_X = empty_X
        vec_X[1::p] = prev_pred[:,h]

        prev_X = np.zeros((M,k))
        for m in range(M):
            total_lags = M*p+1
            prev_X[m,m*(total_lags):(m+1)*total_lags] = vec_X

    return prev_pred