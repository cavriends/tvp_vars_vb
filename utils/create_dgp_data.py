def create_dgp_data():

    # Create data
    lagged_y = np.zeros((T,M*p))

    locations = np.random.randint(0,10,size=M)
    position_counter = 0

    for m in range(M):
        y_m = np.random.normal(loc=locations[m], scale=2, size=T+p)
        for i in range(1,p+1):
            lagged_y[:,position_counter] = y_m[(p-i):-i]
            position_counter += 1

    # Create lagged dependent matrix   
    X = np.zeros((T,M,k))
    stacked_X = np.zeros((M,T,k))

    for m in range(M):
        total_lags = M*p
        stacked_X[m,:,m*(total_lags):(m+1)*total_lags] = lagged_y

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

    beta = np.zeros((T,k))

    for i in range(k):
        bound = np.random.randint(lb,ub)

        if sign == 1:
            sign = -1
        else:
            sign = 1

        beta[:,i] = np.linspace(bound,bound+sign*difference,T) + np.random.normal(scale=scale,size=T)

        #beta[:,27:32] = 0
        beta[50:125,2] = 0

    # Construct dependent
    y = np.zeros((M,T))

    for i in range(T):
        y[:,i] = X[i]@beta[i] + np.random.normal(size=M).T #np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag([0.10,5]))

    # Transpose for KF
    y = y.T
    
    return X, y