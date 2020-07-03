import time
import numpy as np
from IPython.display import clear_output
from scipy.stats import norm

class TVPVARModel:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.T = X.shape[0]
        self.M = X.shape[1]
        self.k = X.shape[2]
        self.initialized_priors = False
        self.initialized_volatility = False

    def initialize_priors(self, prior='svss', prior_parameters={'tau_0':0.1, 'tau_1':10, 'pi0':0.5},
                          prior_default=True):
        self.prior = prior

        if self.prior == 'svss':
            if prior_default:
                prior_parameters = {'tau_0':0.1, 'tau_1':10, 'pi0':0.5}

            self.tau_0 = prior_parameters['tau_0']
            self.tau_1 = prior_parameters['tau_1']
            self.pi0 = prior_parameters['pi0']

        elif self.prior == 'horseshoe':
            if prior_default:
                prior_parameters = {'a0': 10, 'b0': 10}

            self.a0_horseshoe = prior_parameters['a0']
            self.b0_horseshoe = prior_parameters['b0']
            self.lambda_t_horseshoe = np.ones(self.T)
            self.phi_t = np.ones((self.T, self.k))

        elif prior == 'lasso':
            if prior_default:
                prior_parameters = {'lambda_param': 50}

            self.tau_lasso = np.ones((self.T, self.k))
            self.lambda_param = prior_parameters['lambda_param']

        self.initialized_priors = True

    def set_volatility(self, homoskedastic=True):

        self.homoskedastic = homoskedastic

        if homoskedastic:
            # sigma ~ Gamma(at, bt)
            self.at_h = np.ones(self.M)
            self.a0_h = 1
            self.bt_h = np.ones(self.M)
            self.b0_h = 1
        else:
            # sigma_t ~ Gamma(at,bt)
            self.at = np.ones((self.T, self.M))
            self.a0 = 1e-2
            self.bt = np.ones((self.T, self.M))
            self.b0 = 1e-2

        self.initialized_volatility = True


    def train(self, iterations=1000, threshold=1.0e-4, print_status=True):

        if not self.initialized_priors:
            self.initialize_priors()
            
        if not self.initialized_volatility:
            self.set_volatility()

        # beta_0 ~ N(m0,S0)
        m0 = np.zeros(self.k)
        S0 = 4*np.eye(self.k)

        # q_t ~ Gamma(ct,dt)
        ct = np.ones((self.T,self.k))
        dt = np.ones((self.T,self.k))
        d0 = 1
        c0 = 25

        self.tv_probs = np.ones((self.T, self.k))
        self.sigma_t = 0.1 * np.ones((self.T, self.M))

        mtt  = np.zeros((self.k,self.T))
        self.mt1t = np.zeros((self.k,self.T))
        mtt1 = np.zeros((self.k,self.T))
        Stt1 = np.zeros((self.k,self.k,self.T))
        Stt  = np.zeros((self.k,self.k,self.T))

        lambda_t = np.zeros((self.T,self.k))
        q_t      = np.ones((self.T,self.k))
        Qtilde   = np.zeros((self.k,self.k,self.T))
        Ftilde   = np.zeros((self.k,self.k,self.T))
    
        offset    = 0.0015
        delta     = 0.9

        elapsed_time = 0
        start_time = 0
        counter = 0
        mt1t_previous = np.ones((self.k,self.T))
        difference_parameters = np.zeros(iterations)

        while (counter < iterations) & (np.linalg.norm(self.mt1t - mt1t_previous) > threshold):

            difference_parameters[counter] = np.linalg.norm(self.mt1t - mt1t_previous)
            mt1t_previous = self.mt1t
            start_iteration = time.time()

            if print_status:
                if (counter % 10) == 0:
                    if counter != 0:
                        elapsed_time = time.time() - start_time

                    start_time = time.time()
                    clear_output(wait=True)
                    print("Iteration: " + str(counter) + "\n" + "Elapsed time: " + str(elapsed_time) + " seconds")

                if counter == iterations:
                    clear_output(wait=True)
                    print("Done!")

            self.mt1t = np.zeros((self.k,self.T))
            self.St1t = np.zeros((self.k, self.k, self.T))

            # Kalman filter
            # ==================| Update \beta_{t} using Kalman filter/smoother
            for t in range(self.T):
                if self.prior == 'none':
                    Qtilde[:,:,t]  = np.diag(1/(q_t[t,:]))
                    Ftilde[:,:,t]  = np.eye(self.k)
                elif self.prior == 'svss':
                    Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + lambda_t[t,:]))
                    Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))
                elif self.prior == 'horseshoe':
                    Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + self.lambda_t_horseshoe[t]*self.phi_t[t,:]))
                    Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))
                elif self.prior == 'lasso':
                    Qtilde[:,:,t]  = np.diag(1/(q_t[t,:] + self.tau_lasso[t,:]))
                    Ftilde[:,:,t]  = np.multiply(Qtilde[:,:,t],np.diag(q_t[t,:]))

                if t==0:
                    mtt1[:,t]   = Ftilde[:,:,t]@m0
                    Stt1[:,:,t] = Ftilde[:,:,t]@S0@Ftilde[:,:,t].T
                else:
                    mtt1[:,t]   = Ftilde[:,:,t]@mtt[:,t-1]
                    Stt1[:,:,t] = Ftilde[:,:,t]@Stt[:,:,t-1]@Ftilde[:,:,t].T + Qtilde[:,:,t]

                Sx              = Stt1[:,:,t]@self.X[t,:].T
                Kt              = Sx@np.linalg.inv((self.X[t,:] @ Sx + self.sigma_t[t, :]))
                mtt[:,t]        = mtt1[:,t] + Kt@(self.y[t,:] - self.X[t,:]@mtt1[:,t])
                Stt[:,:,t]      = (np.eye(self.k) - Kt@self.X[t,:])@Stt1[:,:,t]

                # Fixed interval smoother
                self.mt1t[:,t] = mtt[:,t]
                self.St1t[:, :, t] = Stt[:, :, t]

            for t in reversed(range(self.T-1)):
                C = (Stt[:,:,t]@Ftilde[:,:,t+1])@np.linalg.inv(Stt1[:,:,t+1])
                self.mt1t[:,t]   = mtt[:,t] + C@(self.mt1t[:,t+1] - mtt1[:,t+1])
                self.St1t[:, :, t] = Stt[:, :, t] + C @ (self.St1t[:, :, t + 1] - Stt1[:, :, t + 1]) @ C.T

            for t in range(self.T):
                eyeF = (np.eye(self.k) - 2*Ftilde[:,:,t]).T
                if t == 0:
                    D = self.St1t[:, :, t] + self.mt1t[:, t] @ self.mt1t[:, t].T + (S0 + m0 * m0.T) @ eyeF
                else:
                    D = self.St1t[:, :, t] + self.mt1t[:, t] @ self.mt1t[:, t].T + (
                                self.St1t[:, :, t - 1] + self.mt1t[:, t - 1] @ self.mt1t[:, t - 1].T) @ eyeF

                # State variances Q_{t}
                ct[t,:]     = c0 + 0.5
                dt[t,:]     = d0 + np.maximum(1e-10,np.diag(D)/2)
                q_t[t,:]    = ct[t,:]/dt[t,:]

            for t in range(self.T):
                if self.prior == 'svss':
                    l_0           = norm.logpdf(self.mt1t[:,t],np.zeros(self.k),self.tau_0*np.ones(self.k))
                    l_1           = norm.logpdf(self.mt1t[:,t],np.zeros(self.k),self.tau_1*np.ones(self.k))
                    gamma         = 1/(np.multiply(1+(np.divide((1-self.pi0),self.pi0)),np.exp(l_0-l_1)))
                    self.pi0      = np.mean(gamma)
                    self.tv_probs[t, :] = gamma
                    lambda_t[t,:] = (1/(self.tau_0**2))*np.ones(self.k)
                    lambda_t[t,gamma==1] = (1/(self.tau_1**2))

                elif self.prior == 'horseshoe':
                    self.lambda_t_horseshoe[t] = (self.a0_horseshoe+1/2*((1/self.phi_t[t,:])@self.mt1t[:,t]**2))/self.k
                    self.phi_t[t,:] = self.b0_horseshoe+(1/self.lambda_t_horseshoe[t])*self.mt1t[:,t]**2

                elif self.prior == 'lasso':
                    self.tau_lasso[t,:] = 1/np.sqrt((self.lambda_param**2/self.mt1t[:,t]**2))

            # Update volatilities
            if self.homoskedastic:
                for m in range(self.M):

                    self.at_h[m] = self.a0_h + self.T

                    for t in range(self.T):
                        updated_b = np.sum(np.power(self.y[t,m]-self.X[t,m].T@self.mt1t[:,t],2))

                    self.bt_h[m] = self.b0_h + updated_b/2

                    self.sigma_t[:, m] = self.bt_h[m] / self.at_h[m]

            else:
                s_tinv = np.zeros((self.T,self.M))
                for t in range(self.T):
                    temp = self.X[t,:] @ (self.mt1t[:,t] @self.mt1t[:,t].T + self.St1t[:, :, t]) @ self.X[t, :].T - \
                           2 * self.X[t, :] @ self.mt1t[:, t] @ self.y[t, :] + \
                           (1 + offset) * self.y[t, :] @ self.y[t, :].T

                    if t == 0:
                        self.at[t,:] = self.a0 + 0.5
                        self.bt[t,:] = self.b0 + temp[0]/2
                    else:
                        self.at[t,:] = delta*self.at[t-1,:] + 0.5
                        self.bt[t,:] = delta*self.bt[t-1,:] + temp[0]/2

                    s_tinv[t,:] = np.divide(self.at[t,:],self.bt[t,:])

                # Smooth volatilities
                phi = np.zeros((self.T,self.M))
                phi[self.T-1,:] = np.divide(self.at[self.T-1,:],self.bt[self.T-1,:])
                for t in reversed(range(self.T-1)):
                    phi[t,:] = [1-delta]*s_tinv[t,:] + delta*phi[t+1,:]
                self.sigma_t = 1 / phi

            if print_status:
                end_iteration = time.time()
                iteration_delta = end_iteration - start_iteration
                print("Seconds for one iteration: " + str(iteration_delta) +
                      "\n" + "Difference: " + str(difference_parameters[counter]))
            # Increase counter
            counter += 1

        return self.mt1t, self.St1t