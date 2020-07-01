def tvp_var_vb(X, y, T, M, p, k, prior='svss', prior_parameters = {'tau_0':0.1, 'tau_1':10, 'pi0':0.5}, prior_default=True, homoskedastic=True, iterations=1000, threshold=1.0e-4, print_status=True):
    
    #Priors
    # 1) beta_0 ~ N(m0,S0)
    m0 = np.zeros(k) 
    S0 = 4*np.eye(k)
    
    # 2) q_t ~ Gamma(ct,dt)
    ct = np.ones((T,k)) 
    dt = np.ones((T,k))  
    d0 = 1
    c0 = 25

    if prior == 'svss':
        # 3) SVSS
        # Default {'tau_0':0.1, 'tau_1':10, 'pi0':0.5}
        if prior_default:
            prior_parameters = {'tau_0':0.1, 'tau_1':10, 'pi0':0.5}
            
        tau_0 = prior_parameters['tau_0']
        tau_1 = prior_parameters['tau_1']
        pi0 = prior_parameters['pi0']
        tv_probs = np.ones((T,k))
    elif prior == 'horseshoe':
        # 4) Horseshoe 
        # Default: {'a0':10,'b0':10}
        if prior_default:
            prior_parameters = {'a0':10,'b0':10}
            
        a0_horseshoe = prior_parameters['a0']
        b0_horseshoe = prior_parameters['b0']
        lambda_t_horseshoe = np.ones(T)
        phi_t = np.ones((T,k))
    elif prior == 'lasso':
        # 5) Lasso
        # Default {'lambda_param':50}
        if prior_default:
            prior_parameters = {'lambda_param':50}
        tau_lasso = np.ones((T,k))
        lambda_param = prior_parameters['lambda_param']
        
    if homoskedastic:
        # 6) sigma ~ Gamma(at, bt) - homoskedastic
        at_h = np.ones(M)
        a0_h = 1
        bt_h = np.ones(M)
        b0_h = 1
    else:
        # 7) sigma_t ~ Gamma(at,bt) - heteroskedastic
        at = np.ones((T,M))
        a0 = 1e-2
        bt = np.ones((T,M))
        b0 = 1e-2
    
    sigma_t = 0.1*np.ones((T,M))

    mtt  = np.zeros((k,T))
    mt1t = np.zeros((k,T))
    mtt1 = np.zeros((k,T))
    Stt  = np.zeros((k,k,T))
    Stt1 = np.zeros((k,k,T))

    lambda_t = np.zeros((T,k))
    q_t      = np.ones((T,k))
    Qtilde   = np.zeros((k,k,T))
    Ftilde   = np.zeros((k,k,T))
    
    offset    = 0.0015
    delta     = 0.9

    elapsed_time = 0
    start_time = 0
    counter = 0
    mt1t_previous = np.ones((k,T))
    difference_parameters = np.zeros(iterations)

    while (counter < iterations) & (np.linalg.norm(mt1t - mt1t_previous) > threshold):

        difference_parameters[counter] = np.linalg.norm(mt1t - mt1t_previous)
        mt1t_previous = mt1t
        start_iteration = time.time()
    
        if print_status:
            if (counter % 10) == 0:
                if counter != 0:
                    elapsed_time = time.time() - start_time

                start_time = time.time()
                clear_output(wait=True)
                print("Iteration: " + str(counter) + "\n" + "Elapsed time: " + str(elapsed_time) + " seconds")

            if (counter == iterations):
                clear_output(wait=True)
                print("Done!")

        # Kalman filter
        # ==================| Update \beta_{t} using Kalman filter/smoother
        for t in range(T):
            if prior == 'none':
                Qtilde[:,:,t]  = np.diag(1/(q_t[t,:]))
                Ftilde[:,:,t]  = np.eye(k)
            elif prior == 'svss':
                Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + lambda_t[t,:]))           
                Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))
            elif prior == 'horseshoe':
                Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + lambda_t_horseshoe[t]*phi_t[t,:]))           
                Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))
            elif prior == 'lasso':
                Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + tau_lasso[t,:]))           
                Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))
                
            if t==0:
                mtt1[:,t]   = Ftilde[:,:,t]@m0;               
                Stt1[:,:,t] = Ftilde[:,:,t]@S0@Ftilde[:,:,t].T
            else:
                mtt1[:,t]   = Ftilde[:,:,t]@mtt[:,t-1]
                Stt1[:,:,t] = Ftilde[:,:,t]@Stt[:,:,t-1]@Ftilde[:,:,t].T + Qtilde[:,:,t]

            Sx              = Stt1[:,:,t]@X[t,:].T        
            Kt              = Sx@np.linalg.inv((X[t,:]@Sx + sigma_t[t,:]))
            mtt[:,t]        = mtt1[:,t] + Kt@(y[t,:] - X[t,:]@mtt1[:,t])
            Stt[:,:,t]      = (np.eye(k) - Kt@X[t,:])@Stt1[:,:,t]
        
        # Fixed interval smoother    
        mt1t = np.zeros((k,T)) 
        St1t = np.zeros((k,k,T))
        mt1t[:,t] = mtt[:,t]
        St1t[:,:,t] = Stt[:,:,t]

        for t in reversed(range(T-1)):
            C = (Stt[:,:,t]@Ftilde[:,:,t+1])@np.linalg.inv(Stt1[:,:,t+1])        
            mt1t[:,t]   = mtt[:,t] + C@(mt1t[:,t+1] - mtt1[:,t+1]) 
            St1t[:,:,t] = Stt[:,:,t] + C@(St1t[:,:,t+1] - Stt1[:,:,t+1])@C.T
            
        if np.isnan(mt1t).all():
            print("Fucked up")

        for t in range(T):
            eyeF = (np.eye(k) - 2*Ftilde[:,:,t]).T
            if t == 0:
                D = St1t[:,:,t] + mt1t[:,t]@mt1t[:,t].T + (S0 + m0*m0.T)@eyeF
            else:
                D = St1t[:,:,t] + mt1t[:,t]@mt1t[:,t].T + (St1t[:,:,t-1] + mt1t[:,t-1]@mt1t[:,t-1].T)@eyeF

            # State variances Q_{t}
            ct[t,:]     = c0 + 0.5
            dt[t,:]     = d0 + np.maximum(1e-10,np.diag(D)/2)
            q_t[t,:]    = ct[t,:]/dt[t,:]
            
        for t in range(T):
            if prior == 'svss':
                l_0           = norm.logpdf(mt1t[:,t],np.zeros(k),tau_0*np.ones(k))
                l_1           = norm.logpdf(mt1t[:,t],np.zeros(k),tau_1*np.ones(k))
                gamma         = 1/(np.multiply(1+(np.divide((1-pi0),pi0)),np.exp(l_0-l_1)))
                pi0           = np.mean(gamma)
                tv_probs[t,:] = gamma
                lambda_t[t,:] = (1/(tau_0**2))*np.ones(k)
                lambda_t[t,gamma==1] = (1/(tau_1**2))
                
            elif prior == 'horseshoe':
                lambda_t_horseshoe[t] = (a0_horseshoe+1/2*((1/phi_t[t,:])@mt1t[:,t]**2))/k
                phi_t[t,:] = b0_horseshoe+(1/lambda_t_horseshoe[t])*mt1t[:,t]**2
                
            elif prior == 'lasso':
                tau_lasso[t,:] = 1/np.sqrt((lambda_param**2/mt1t[:,t]**2))

        # Update volatilities
        
        if homoskedastic:
            for m in range(M):

                at_h[m] = a0_h + T

                for t in range(T):
                    updated_b = np.sum(np.power(y[t,m]-X[t,m].T@mt1t[:,t],2))

                bt_h[m] = b0_h + updated_b/2

                sigma_t[:,m] = bt_h[m]/at_h[m]
        
        if homoskedastic == False:
            s_tinv = np.zeros((T,M));
            for t in range(T):           
                temp = X[t,:]@(mt1t[:,t]@mt1t[:,t].T + St1t[:,:,t])@X[t,:].T - 2*X[t,:]@mt1t[:,t]@y[t,:] + (1 + offset)*y[t,:]@y[t,:].T;        
                if t == 0:
                    at[t,:] = a0 + 0.5;
                    bt[t,:] = b0 + temp[0]/2;
                else:
                    at[t,:] = delta*at[t-1,:] + 0.5;
                    bt[t,:] = delta*bt[t-1,:] + temp[0]/2;

                s_tinv[t,:] = np.divide(at[t,:],bt[t,:]);

            # Smooth volatilities
            phi = np.zeros((T,M)); 
            phi[T-1,:] = np.divide(at[T-1,:],bt[T-1,:]);
            for t in reversed(range(T-1)):
                phi[t,:] = [1-delta]*s_tinv[t,:] + delta*phi[t+1,:];
            sigma_t = 1/phi;
        
        if print_status:
            end_iteration = time.time()
            iteration_delta = end_iteration - start_iteration
            print("Seconds for one iteration: " + str(iteration_delta) + "\n" + "Difference: " + str(difference_parameters[counter]))
        # Increase counter
        counter += 1
        
    return mt1t, St1t