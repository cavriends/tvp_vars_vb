def create_data(series, train_indices, T, M, p, k):
    
    # Create data
    lagged_y = np.ones((T,M*p+1))
    lagged_series = []
    y_series = []
    
    for s in series:
        lagged_series.append(s[:train_indices])

    position_counter = 1 #Constant is added in front

    for m in range(M):
        y_m = lagged_series[m]
        for i in range(1,p+1):
            lagged_y[:,position_counter] = y_m[(p-i):-i]
            position_counter += 1

    # Create lagged dependent matrix   
    X = np.zeros((T,M,k))
    stacked_X = np.zeros((M,T,k))

    for m in range(M):
        total_lags = M*p+1
        stacked_X[m,:,m*(total_lags):(m+1)*total_lags] = lagged_y

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