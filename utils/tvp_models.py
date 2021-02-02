import time
import numpy as np
from IPython.display import clear_output
from scipy.stats import norm, multivariate_normal
from scipy.special import psi, gammaln
from datetime import datetime, timedelta


# This class is the backbone of the VI-based TVP-BVAR. The final class that is the complete model as in the thesis is
# the tvp_ar_non_contemp(). However, the complete estimation algorithm is in the TVPVARModel class. A few notes if one
# uses this model. The 'lasso-alternative' prior is the actual Lasso prior used in the paper. Furthermore, my preference
# is to keep some space between the lines to enhance readability, this is not in line with common practice, I know.
# In addition, this code is a port to Python from the MATLAB code of Koop & Korobilis (2020)
# "Bayesian dynamic variable selection in high dimensions".

class TVPVARModel:

    def __init__(self, X, y, p, train_index, constant=True, iterations=100, homoskedastic=True):
        self.X = X
        self.y = y
        self.T = X.shape[0]
        self.M = X.shape[1]
        if constant:
            self.k = self.M*(self.M*p+1)
        else:
            self.k = self.M*(self.M*p)
        self.p = p
        self.iterations = iterations
        self.train_index = train_index
        self.T_train = train_index
        self.initialized_priors = False
        self.initialized_volatility = False
        self.prior_parameters = None
        self.prediction_switch = False
        self.print_status = False
        self.homoskedastic = homoskedastic

    def create_train(self):
        self.X_train = self.X[:self.T_train, :]
        self.y_train = self.y[:self.T_train, :]

    def initialize_priors(self, prior='svss', prior_parameters=None):

        ## Initialize the priors, if no prior parameters are provided, fall back to default values.
        self.prior = prior
        prior_default = True

        if prior_parameters != None:
            self.prior_parameters = prior_parameters
            prior_default = False

        if self.prior == 'svss':
            if prior_default:
                self.prior_parameters = {'g0': 1, 'h0': 12, 'pi0': 0.5}

            self.gt = np.ones((self.T,self.k))
            self.g0 = self.prior_parameters['g0']
            self.ht = np.ones((self.T,self.k))
            self.h0 = self.prior_parameters['h0']
            self.tau_1 = self.gt/self.ht
            self.cons = 1e-4
            self.tau_0 = self.cons * self.tau_1
            self.pi0 = self.prior_parameters['pi0'] * np.ones((self.T,1))
            self.tv_probs = np.ones((self.T,self.k))

        elif self.prior == 'horseshoe':
            if prior_default:
                # self.prior_parameters = {'a0': 1/2, 'b0': 1}
                self.prior_parameters = {'a0': 10, 'b0': 10}

            self.a0_horseshoe = self.prior_parameters['a0']
            self.b0_horseshoe = self.prior_parameters['b0']
            self.lambda_t_horseshoe = np.ones(self.T_train)
            self.phi_t = np.ones((self.T_train, self.k))
            self.v = np.ones(((self.T_train, self.k)))
            self.delta = np.ones(self.T_train)

        elif prior == 'lasso':
            if prior_default:
                self.prior_parameters = {'lambda_param': 50}

            self.tau_lasso = np.ones((self.T_train, self.k))
            self.lambda_param = self.prior_parameters['lambda_param']

        elif prior == 'lasso_alternative':
            if prior_default:
                self.prior_parameters = {'a0_lasso': 1e-3, 'b0_lasso': 1e-3}

            self.tau_lasso = np.ones((self.T_train, self.k))
            self.a0_lasso = np.ones(self.T)*self.prior_parameters["a0_lasso"]
            self.b0_lasso = np.ones(self.T) * self.prior_parameters["b0_lasso"]
            self.lambda_lasso = np.ones(self.T)

        self.initialized_priors = True

    def set_volatility(self):

        ## Set the priors for the volatility (if homoskedastic or heteroskedastic)

        if self.homoskedastic:
            # sigma ~ Gamma(at, bt)
            self.at_h = np.ones(self.M)
            self.a0_h = 1
            self.bt_h = np.ones(self.M)
            self.b0_h = 1
        else:
            # sigma_t ~ Gamma(at,bt)
            self.at = np.ones((self.T_train, self.M))
            self.a0 = 1e-2
            self.bt = np.ones((self.T_train, self.M))
            self.b0 = 1e-2
            self.delta_variance = 0.90

        self.initialized_volatility = True

    def train(self, threshold=1.0e-4, print_status=True, kl_bound=False):

        ## The procedure to estimate a TVP-BVAR using VI with one of the three priors

        self.create_train()

        ## Initialize priors if they're not initialized

        if (not self.initialized_priors) & (not self.prediction_switch):
            self.initialize_priors()
        elif (not self.initialized_priors) & self.prediction_switch:
            self.initialize_priors(prior=self.prior, prior_parameters=self.prior_parameters)

        if not self.initialized_volatility:
            self.set_volatility()

        ## Set priors for beta_0 and q_t. Furthermore, initialize all necessary matrices and vectors

        # beta_0 ~ N(m0,S0)
        m0 = np.zeros(self.k)
        S0 = 4 * np.eye(self.k)

        # q_t ~ Gamma(ct,dt)
        ct = np.ones((self.T_train, self.k))
        dt = np.ones((self.T_train, self.k))
        d0 = 1
        c0 = 100

        self.D = np.zeros((self.T_train, self.k, self.k))
        self.tv_probs = np.ones((self.T_train, self.k))
        self.sigma_t = 0.1 * np.ones((self.T_train, self.M))

        mtt = np.zeros((self.k, self.T_train))
        self.mt1t = np.zeros((self.k, self.T_train))
        mtt1 = np.zeros((self.k, self.T_train))
        Stt1 = np.zeros((self.k, self.k, self.T_train))
        Stt = np.zeros((self.k, self.k, self.T_train))

        lambda_t = np.zeros((self.T_train, self.k))
        q_t = np.ones((self.T_train, self.k))
        Qtilde = np.zeros((self.k, self.k, self.T_train))
        Ftilde = np.zeros((self.k, self.k, self.T_train))

        self.F_new = 0
        self.F_old = 1
        self.KL = np.ones(self.iterations)*12345
        offset = 0.0015
        delta = 0.8

        ## Initialize variables for timing, counting and convergence threshold
        elapsed_time = 0
        start_time = 0
        self.counter = 0
        mt1t_previous = np.ones((self.k, self.T_train))
        difference_parameters = np.zeros(self.iterations)

        ## The VI algorithm is not converged until the difference between parameters is lower than the threshold
        ## Or the iteration limit is reached

        while (self.counter < self.iterations) & (np.linalg.norm(self.mt1t - mt1t_previous) > threshold) & (np.abs(self.F_new - self.F_old) > threshold):

            difference_parameters[self.counter] = np.linalg.norm(self.mt1t - mt1t_previous)
            mt1t_previous = self.mt1t
            start_iteration = time.time()

            ## If print_status = True, the progress of the algorithm is printed
            if print_status:
                if (self.counter % 10) == 0:
                    if self.counter != 0:
                        elapsed_time = time.time() - start_time

                    start_time = time.time()
                    clear_output(wait=True)
                    print(f"Iteration: {self.counter} \n Elapsed time: {np.round(elapsed_time,4)} seconds")

                if self.counter == self.iterations:
                    clear_output(wait=True)
                    print("Done!")

            self.mt1t = np.zeros((self.k, self.T_train))
            self.St1t = np.zeros((self.k, self.k, self.T_train))


            ## The Kalman filter is initialized for the different priors.
            # Kalman filter
            # ==================| Update \beta_{t} using Kalman filter/smoother
            for t in range(self.T_train):
                if self.prior == 'none':
                    Qtilde[:, :, t] = np.diag(1 / (q_t[t, :]))
                    Ftilde[:, :, t] = np.eye(self.k)
                elif self.prior == 'svss':
                    Qtilde[:, :, t] = np.diag(1 / (q_t[t, :] + lambda_t[t, :]))
                    Ftilde[:, :, t] = np.multiply(Qtilde[:, :, t], np.diag(q_t[t, :]))
                elif self.prior == 'horseshoe':
                    Qtilde[:, :, t] = np.diag(1 / (q_t[t, :] + self.lambda_t_horseshoe[t] * self.phi_t[t, :]))
                    Ftilde[:, :, t] = np.multiply(Qtilde[:, :, t], np.diag(q_t[t, :]))
                elif self.prior == 'lasso':
                    Qtilde[:, :, t] = np.diag(1 / (q_t[t, :] + self.tau_lasso[t, :]))
                    Ftilde[:, :, t] = np.multiply(Qtilde[:, :, t], np.diag(q_t[t, :]))
                elif self.prior == 'lasso_alternative':
                    Qtilde[:, :, t] = np.diag(1 / (q_t[t, :] + self.tau_lasso[t, :]))
                    Ftilde[:, :, t] = np.multiply(Qtilde[:, :, t], np.diag(q_t[t, :]))

                if t == 0:
                    mtt1[:, t] = Ftilde[:, :, t] @ m0
                    Stt1[:, :, t] = Ftilde[:, :, t] @ S0 @ Ftilde[:, :, t].T
                else:
                    mtt1[:, t] = Ftilde[:, :, t] @ mtt[:, t - 1]
                    Stt1[:, :, t] = Ftilde[:, :, t] @ Stt[:, :, t - 1] @ Ftilde[:, :, t].T + Qtilde[:, :, t]

                Sx = Stt1[:, :, t] @ self.X_train[t, :].T
                if self.k == 1:
                    Kt = Sx * (1/(self.X_train[t, :] @ Sx + self.sigma_t[t, :]))
                else:
                    Kt = Sx @ np.linalg.inv((self.X_train[t, :] @ Sx + self.sigma_t[t, :]))
                mtt[:, t] = mtt1[:, t] + Kt @ (self.y_train[t, :] - self.X_train[t, :] @ mtt1[:, t])
                Stt[:, :, t] = (np.eye(self.k) - Kt @ self.X_train[t, :]) @ Stt1[:, :, t]

                # Fixed interval smoother
                self.mt1t[:, t] = mtt[:, t]
                self.St1t[:, :, t] = Stt[:, :, t]

            for t in reversed(range(self.T_train - 1)):
                C = (Stt[:, :, t] @ Ftilde[:, :, t + 1]) @ np.linalg.inv(Stt1[:, :, t + 1])
                self.mt1t[:, t] = mtt[:, t] + C @ (self.mt1t[:, t + 1] - mtt1[:, t + 1])
                self.St1t[:, :, t] = Stt[:, :, t] + C @ (self.St1t[:, :, t + 1] - Stt1[:, :, t + 1]) @ C.T

            for t in range(self.T_train):
                eyeF = (np.eye(self.k) - 2 * Ftilde[:, :, t]).T
                if t == 0:
                    self.D[t,:,:] = self.St1t[:, :, t] + self.mt1t[:, t] @ self.mt1t[:, t].T + (S0 + m0 * m0.T) @ eyeF
                else:
                    self.D[t,:,:] = self.St1t[:, :, t] + self.mt1t[:, t] @ self.mt1t[:, t].T + (
                                    self.St1t[:, :, t - 1] + self.mt1t[:, t - 1] @ self.mt1t[:, t - 1].T) @ eyeF

                # State variances Q_{t}
                ct[t, :] = c0 + 0.5
                dt[t, :] = d0 + np.maximum(1e-10, np.diag(self.D[t,:,:]) / 2)
                q_t[t, :] = ct[t, :] / dt[t, :]

            ## The variational posteriors are updated for each of the priors, SVSS, Horseshoe or Lasso
            for t in range(self.T_train):
                if self.prior == 'svss':
                    self.gt[t,:] = self.g0 + 0.5
                    self.ht[t,:] = self.h0 + (self.mt1t[:, t]**2)/2
                    self.tau_1[t,:]  = self.ht[t,:]/self.gt[t,:]
                    self.tau_0[t,:]  = self.cons * self.tau_1[t,:]
                    self.l_0 = norm.logpdf(self.mt1t[:, t], np.zeros(self.k), np.sqrt(self.tau_0[t,:].T)) + 1e-20
                    self.l_1 = norm.logpdf(self.mt1t[:, t], np.zeros(self.k), np.sqrt(self.tau_1[t,:].T)) + 1e-20
                    self.gamma = 1/(1 + ((1 - self.pi0[t])/self.pi0[t])*np.exp(self.l_0 - self.l_1))
                    self.pi0[t] = (1 + self.gamma[self.gamma==1].sum())/(2 + self.k)
                    self.tv_probs[t,:] = self.gamma
                    lambda_t[t,:] = 1/(((1 - self.gamma)**2)*self.tau_0[t,:].T + (self.gamma**2)*self.tau_1[t,:].T)

                elif self.prior == 'horseshoe':
                    self.lambda_t_horseshoe[t] = (1/self.delta[t] + 0.5 * (
                                                 (1 / self.phi_t[t, :]) @ (self.mt1t[:, t] ** 2 +
                                                  np.abs(np.diag(self.St1t[:, :, t]))))) / (self.k + 1/2 - 1)

                    self.phi_t[t, :] = (1/self.v[t,:] +
                                        (self.mt1t[:, t] ** 2 + np.abs(np.diag(self.St1t[:, :, t])))
                                        / 2*self.lambda_t_horseshoe[t])/0.5

                    self.v[t,:] = (self.b0_horseshoe + (1/(self.phi_t[t,:])**2 + np.abs(np.diag(self.St1t[:, :, t]))))/(self.a0_horseshoe - 1)
                    self.delta[t] = (self.b0_horseshoe + (1/self.lambda_t_horseshoe[t]))/(self.a0_horseshoe - 1)

                elif self.prior == 'lasso':
                    self.tau_lasso[t, :] = 1 / np.sqrt((self.lambda_param ** 2 / self.mt1t[:, t] ** 2))

                elif self.prior == 'lasso_alternative':
                    self.lambda_lasso[t] = (self.k + self.a0_lasso[t])/(np.sum(self.tau_lasso[t,:])/2+self.b0_lasso[t])
                    self.tau_lasso[t, :] = 1 / np.sqrt((self.lambda_lasso[t] ** 2 / self.mt1t[:, t] ** 2))

            if (self.prior == 'horseshoe') & (print_status):
                print(f'lambda: {self.lambda_t_horseshoe.mean()},'
                      f'phi: {self.phi_t.mean()}, v: {self.v.mean()}, delta: {self.delta.mean()} '
                      f'mt1t: {self.mt1t.mean()}')

            # Update volatilities
            if self.homoskedastic:

                for m in range(self.M):

                    self.at_h[m] = self.a0_h + self.T_train

                    for t in range(self.T_train):

                        if self.k == 1:
                            updated_b = self.b0_h + np.sum(np.power(self.y_train[t, m] - self.X_train[t, m].T * self.mt1t[:, t], 2) + self.X_train[t, m].T * self.St1t[:,:,t] * self.X_train[t, m])
                        else:
                            updated_b = self.b0_h + np.sum(
                                        np.power(self.y_train[t, m] - self.X_train[t, m].T @ self.mt1t[:, t], 2)
                                        + self.X_train[t, m].T @ self.St1t[:, :, t] @ self.X_train[t, m])

                    self.bt_h[m] = self.b0_h + updated_b / 2

                    self.sigma_t[:,m] = self.bt_h[m] / (self.at_h[m]-1)

            else:
                s_tinv = np.zeros((self.T_train, self.M))
                for t in range(self.T_train):
                    temp = self.X_train[t, :] @ (
                                self.mt1t[:, t] @ self.mt1t[:, t].T + self.St1t[:, :, t]) @ self.X_train[t, :].T - \
                           2 * self.X_train[t, :] @ self.mt1t[:, t] @ self.y_train[t, :] + \
                           (1 + offset) * self.y_train[t, :] @ self.y_train[t, :].T

                    if t == 0:
                        self.at[t, :] = self.a0 + 0.5
                        self.bt[t, :] = self.b0 + temp[0] / 2
                    else:
                        self.at[t, :] = self.delta_variance * self.at[t - 1, :] + 0.5
                        self.bt[t, :] = self.delta_variance * self.bt[t - 1, :] + temp[0] / 2

                    s_tinv[t, :] = np.divide(self.at[t, :], self.bt[t, :])

                # Smooth volatilities
                phi = np.zeros((self.T_train, self.M))
                phi[self.T_train - 1, :] = np.divide(self.at[self.T_train - 1, :], self.bt[self.T_train - 1, :])
                for t in reversed(range(self.T_train - 1)):
                    phi[t, :] = [1 - delta] * s_tinv[t, :] + delta * phi[t + 1, :]
                self.sigma_t = 1 / phi


                if kl_bound:
                    # Calculate ELBO
                    self.density = np.empty(self.T_train)
                    for t in range(self.T_train):
                        self.density[t] = norm.logpdf(self.y_train[t,:], self.X_train[t,:]@self.mt1t[:,t], self.sigma_t[t])
                    self.F_old = self.F_new

                    self.density_wo = self.density[~np.isnan(self.density)]

                    logpdf_density = np.linalg.norm(self.density_wo)
                    klgamma_1 = sum(self.klgamma(self.gt,self.ht,self.g0,self.h0))
                    klgamma_2 = sum(self.klgamma(ct,dt,c0,d0))

                    self.F_new = logpdf_density - klgamma_1 - klgamma_2
                    self.KL[self.counter] = self.F_new

            if print_status:
                end_iteration = time.time()
                iteration_delta = end_iteration - start_iteration
                print(f'Seconds for one iteration: {np.round(iteration_delta,4)}'
                      f'\n Difference: {np.round(difference_parameters[self.counter],4)}')

            # Increase counter
            self.counter += 1

        self.initialized_priors = False
        self.initialized_volatility = False

        return self.mt1t, self.St1t, self.sigma_t

    def klgamma(self, pa, pb, qa, qb):

        ## The KL distance for the gamma distribution. It is not used, but ported from the MATLAB code.

        n = max([pb.shape[1], pa.shape[1]])

        if pa.shape[1] == 1:
            pa = pa * np.ones((1, n))
        if pb.shape[1] == 1:
            pb = pb * np.ones((1, n))

        qa = qa * np.ones((1, n))
        qb = qb * np.ones((1, n))

        kl = sum(pa*np.log(pb) - gammaln(pa) - qa*np.log(qb) + gammaln(qa) + (pa - qa)*(psi(pa) - np.log(pb)) - (pb - qb)*pa/pb)

        return kl

    def calculate_oos_predictions(self, total_h=8, constant=True, number_of_draws=0, print_status=True):

        ## Calculate the out-of-sample predictions for the model.

        # To take the constant into account, there might be a situation where this is not necessary
        constant_binary = 0
        if constant:
            constant_binary = 1

        ## Switch is included to reset the priors as prediction is according to an extending window
        self.prediction_switch = True

        self.initial_T_train = self.T_train
        self.initial_train_index = self.train_index

        self.number_of_predictions = self.T - self.T_train
        self.prev_pred = np.zeros((self.number_of_predictions, self.M, total_h))

        for t in range(self.number_of_predictions):

            begin_time = time.time()

            self.train_index = self.initial_train_index + t
            self.T_train = self.train_index - self.p

            prev_X = self.X[self.T_train]

            __, __ = self.train(print_status=self.print_status)

            for h in range(total_h):
                if number_of_draws > 0:
                    self.prev_pred[t, :, h] = np.median(prev_X @ self.mt1t[:, -1] +
                                                        multivariate_normal.rvs(mean=np.zeros(self.M),
                                                                                cov=np.diag(np.sqrt(self.sigma_t)),
                                                                                size=number_of_draws), 0)
                else:
                    self.prev_pred[t, :, h] = prev_X @ self.mt1t[:, -1]

                vec_X = prev_X[0, :(self.M * self.p + 1*constant_binary)]

                empty_X = np.zeros((self.M * self.p + 1*constant_binary))
                empty_X[0] = 1

                for m in range(self.M):
                    empty_X[(m * self.p + 2*constant_binary):((m + 1) * self.p + 1*constant_binary)] \
                        = vec_X[(m * self.p + 1*constant_binary):((m + 1) * self.p)]

                vec_X = empty_X
                vec_X[1*constant_binary::self.p] = self.prev_pred[t, :, h]

                prev_X = np.zeros((self.M, self.k))
                for m in range(self.M):
                    total_lags = self.M * self.p + 1*constant_binary
                    prev_X[m, m * (total_lags):(m + 1) * total_lags] = vec_X

            self.T_train = self.initial_T_train
            self.train_index = self.initial_train_index

            if print_status:
                elapsed_time = time.time() - begin_time
                converged = False
                if self.counter < self.iterations:
                    converged = True
                print(f'Progress: {t + 1}/{self.number_of_predictions} '
                      f'| Elapsed time: {elapsed_time} seconds '
                      f'| Coefficients converged: {converged}')

                print(
                    f'ETA: {datetime.now() + timedelta(seconds=elapsed_time * (self.number_of_predictions - (t + 1)))}')

        return self.prev_pred

    def calculate_metrics(self, total_h=8, number_of_draws=0, constant=True, print_status=True):

        ## Calculate h-step MSFE & ALPL metrics for a pre-specified h-forecast horizon.

        msfe_tvp = np.zeros((total_h, self.M))
        alpl = np.zeros(total_h)

        ## Calculate the out-of-sample predictions that are used to calculate the metrics.

        self.y_pred = self.calculate_oos_predictions(total_h, constant, number_of_draws, print_status)

        for h in range(total_h):

            lpl = np.zeros(self.number_of_predictions - h)

            ## Due to the zero-indexing of Python this is the workaround for the 1-step forecast

            if h == 0:
                y_true_h = self.y[(self.train_index - self.p):]
                y_pred_h = self.y_pred[:, :, 0]

                msfe_tvp[h] = np.mean((y_pred_h - y_true_h) ** 2, 0)
                for t in range(self.number_of_predictions):
                    lpl[t] = np.log(multivariate_normal.pdf(y_true_h[t], y_pred_h[t], cov=np.cov(y_pred_h.T),
                                                     allow_singular=True) + 1e-16)

                alpl[h] = lpl.mean()

            ## For the 2-step until h-step forecast horizons the metrics are calculated below

            else:
                y_true_h = self.y[(self.train_index - self.p + h):]
                y_pred_h = self.y_pred[:-h, :, h]

                msfe_tvp[h] = np.mean((y_pred_h - y_true_h) ** 2, 0)
                for t in range(self.number_of_predictions - h):
                    lpl[t] = np.log(multivariate_normal.pdf(y_true_h[t], y_pred_h[t], cov=np.cov(y_pred_h.T),
                                                     allow_singular=True) + 1e-16)

                alpl[h] = lpl.mean()

        return msfe_tvp, alpl

    def insample_msfe(self):

        ## Estimating insample MSFE

        self.insample_y_pred = np.zeros((self.T_train, self.M))

        for m in range(self.M):

            for t in range(self.T_train):
                self.insample_y_pred[t, m] = self.X_train[t, m, :] @ self.mt1t[:, t]

        self.insample_msfe_calculated = np.mean((self.insample_y_pred - self.y[:self.T_train]) ** 2)

        return self.insample_msfe_calculated

## Outdated TVP-BVAR the contemperanous values approach. The model used in the paper is tvp_ar_contemp_decomposition()

def tvp_ar_contemp(T, M, p, train, X, y, prior='lasso', total_h=8, iterations=50, print_status=False, prior_parameters=None):
    # Contemperous values added

    mt1t_mean_set = []
    sigma_set = []
    msfe_set = []
    alpl_set = []
    y_pred_complete = np.zeros((M,T-train+1,1,total_h))

    # tvp_ar = TVPVARModel(np.expand_dims(X[:, m, (m * (M + M - 1)):(m * (M + M - 1) + (M + M - 1))], 1),
    #                      np.expand_dims(y[m, :].T, 1), p, train, False, homoskedastic=False)

    for m in range(M):
        tvp_ar = TVPVARModel(np.expand_dims(X[:, m], 1),
                             np.expand_dims(y[m, :].T, 1), p, train, False, homoskedastic=False)
        tvp_ar.k = M + M - 1
        tvp_ar.iterations = iterations

        if prior_parameters == None:
            tvp_ar.initialize_priors(prior=prior)
        else:
            tvp_ar.initialize_priors(prior=prior, prior_parameters=prior_parameters)
        mt1t, st1t = tvp_ar.train(print_status=print_status)
        mt1t_mean_set.append(mt1t)
        sigma_set.append(tvp_ar.sigma_t)

        msfe, alpl= tvp_ar.calculate_metrics(total_h, constant=False, print_status=print_status)

        msfe_set.append(msfe)
        alpl_set.append(alpl)

    msfe = np.block(msfe_set).mean(1)
    complete_msfe = np.block(msfe_set)
    alpl = np.block(alpl_set).reshape(total_h, M).mean(1)
    mt1t_full = np.vstack(mt1t_mean_set)
    mt1t_coeff = mt1t_full.reshape((M, M + M - 1, train - 1))[:, :M, :].reshape(M ** 2, train - 1)
    sigma = np.block(sigma_set)

    return msfe, alpl, mt1t_full, mt1t_coeff, sigma, complete_msfe


## Outdated TVP-BVAR without the contemperanous values approach. The model used in the paper is tvp_ar_contemp_decomposition()

def tvp_ar_non_contemp(T, M, p, train, X, y, prior='lasso', total_h=8, iterations=50, print_status=False):
    mt1t_mean_set = []
    sigma_set = []
    msfe_set = []
    alpl_set = []

    for m in range(M):
        tvp_ar = TVPVARModel(np.expand_dims(X[:, 0, 0:M], 1),
                             np.expand_dims(y[m, :].T, 1), p, train, False, homoskedastic=False)
        tvp_ar.k = M
        tvp_ar.iterations = iterations
        tvp_ar.initialize_priors(prior=prior)
        mt1t, st1t = tvp_ar.train(print_status=print_status)
        mt1t_mean_set.append(mt1t)
        sigma_set.append(tvp_ar.sigma_t)

        msfe, alpl, sigma_t = tvp_ar.calculate_metrics(total_h, constant=False, print_status=print_status)

        msfe_set.append(msfe)
        alpl_set.append(alpl)

    #         print(f'Variable: {m+1} | MSFE: {msfe.mean()}')

    msfe = np.block(msfe_set).mean(1)
    alpl = np.block(alpl_set).reshape(total_h, M).mean(1)
    mt1t = np.vstack(mt1t_mean_set)
    sigma = np.block(sigma_set)

    return msfe, alpl, mt1t, sigma

## TVP-BVAR using the Cholesky decomposition approach Huber et. al (2020). It builts on the TVPVARModel class

class tvp_ar_contemp_decomposition:

    def __init__(self, T, M, p, train, X, y, prior='lasso', total_h=8, iterations=50, print_status=False, prior_parameters=None):

        self.T = T
        self.M = M
        self.p = p
        self.train = train
        self.X = X
        self.y = y
        self.prior = prior
        self.total_h = total_h
        self.iterations = iterations
        self.print_status = print_status
        self.prior_parameters = prior_parameters

    def estimate(self, train):

        # Contemperous values added
        mt1t_set = []
        H = np.zeros((train, self.M))

        for m in range(self.M):
            tvp_ar = TVPVARModel(np.expand_dims(self.X[:train, m], 1),
                                 np.expand_dims(self.y[m, :train].T, 1), self.p, train, False, homoskedastic=False)
            tvp_ar.k = self.M * 2
            tvp_ar.iterations = self.iterations

            if self.prior_parameters == None:
                tvp_ar.initialize_priors(prior=self.prior)
            else:
                tvp_ar.initialize_priors(prior=self.prior, prior_parameters=self.prior_parameters)

            mt1t, st1t, sigma_t = tvp_ar.train(print_status=self.print_status)

            mt1t_set.append(mt1t)
            H[:, m] = np.squeeze(sigma_t)

        # Reconstruct the reduced form coefficients
        U_elements = np.zeros(((self.M ** 2 - self.M) // 2, train))
        self.structural_coefficients = np.zeros((self.M, self.M + 1, train))
        self.reduced_coefficients = np.zeros((self.M, self.M + 1, train))
        U = np.empty((self.M, self.M, train))
        self.reduced_covariance = np.zeros((self.M, self.M, train))

        for t in range(train):
            collection_list = []
            coefficent_matrix = np.zeros((self.M, self.M + 1))

            for m in range(self.M):

                coefficent_matrix[m, :] = mt1t_set[m][:(self.M + 1), t]

                if m != 0:
                    collection_list.append(mt1t_set[m][(self.M + 1):(self.M + 1) + m, t])

            U_elements[:, t] = np.block(collection_list)
            self.structural_coefficients[:, :, t] = coefficent_matrix

        for t in range(train):
            id_matrix = np.eye(self.M)
            indices_tril = np.tril_indices(self.M, -1)
            id_matrix[indices_tril] = -U_elements[:, t]
            Sigma = np.diag(H[t, :])
            U_inv = np.linalg.inv(id_matrix)
            U[:, :, t] = U_inv
            self.reduced_coefficients[:, :, t] = U_inv @ self.structural_coefficients[:, :, t]
            self.reduced_covariance[:, :, t] = U_inv @ Sigma @ U_inv.T

        # Make in-sample predictions
        self.y_insample_pred = np.empty((self.M, train))

        for t in range(train):
            self.y_insample_pred[:, t] = self.reduced_coefficients[:, :, t] @ self.X[t, 0, :(self.M + 1)]

    def calculate_forecasts(self, train):

        number_of_predictions = self.T - train

        #Make an iterated out-of-sample forecast
        self.y_pred = np.zeros((number_of_predictions,self.M,self.total_h))

        for idx, t in enumerate(range(train, self.T)):

            for h in range(self.total_h):

                if h == 0:
                    self.y_pred[idx, :, h] = self.reduced_coefficients[:, :, -1] @ np.hstack(
                        (np.ones(1), self.y[:, t]))

                else:
                    self.y_pred[idx, :, h] = self.reduced_coefficients[:, :, -1] @ np.hstack((np.ones(1), self.y_pred[idx, :, h-1]))

        return self.y_pred, self.reduced_coefficients, self.reduced_covariance


    def calculate_metrics(self, train):

        number_of_predictions = self.T - train
        msfe_tvp = np.zeros((self.total_h, self.M))
        alpl = np.zeros(self.total_h)

        self.calculate_forecasts(train)

        for h in range(self.total_h):

            if h < number_of_predictions:
                lpl = np.zeros(number_of_predictions - h)

                if h == 0:
                    y_true_h = self.y.T[train:]
                    y_pred_h = self.y_pred[:, :, 0]

                    msfe_tvp[h] = np.mean((y_pred_h - y_true_h) ** 2, 0)
                    for t in range(number_of_predictions):
                        try:
                            lpl[t] = np.log(multivariate_normal.pdf(y_true_h[t], y_pred_h[t], cov=np.cov(y_pred_h.T),
                                                                allow_singular=True) + 1e-16)
                        except:
                            if self.print_status:
                                print("Covariance might be singular")

                    alpl[h] = lpl.mean()

                else:
                    y_true_h = self.y.T[train + h:]
                    y_pred_h = self.y_pred[:-h, :, h]

                    if y_true_h.shape[0] == 1:
                        covariance = np.cov(y_pred_h)
                    else:
                        covariance = np.cov(y_pred_h.T)

                    msfe_tvp[h] = np.mean((y_pred_h - y_true_h) ** 2, 0)
                    for t in range(number_of_predictions - h):
                        try:
                            lpl[t] = np.log(multivariate_normal.pdf(y_true_h[t], y_pred_h[t], cov=covariance,
                                                                    allow_singular=True) + 1e-16)
                        except:
                            if self.print_status:
                                print(f'Covariance might be singular, in this case it is due to the last observation')

                    alpl[h] = lpl.mean()

        return msfe_tvp, alpl

    def result(self):

        msfe_set = []
        alpl_set = []

        for train_index in range(self.train, self.T):
            self.estimate(train_index)
            msfe, alpl = self.calculate_metrics(train_index)
            msfe_set.append(msfe)
            alpl_set.append(alpl)

        alpl = np.block(alpl_set).reshape(self.T-self.train,self.total_h).mean(0)
        complete_alpl = np.block(alpl_set).reshape(self.T-self.train,self.total_h)
        complete_msfe = np.block(msfe_set)
        msfe = np.block(msfe_set).mean(1)
        coeff = self.reduced_coefficients
        sigma = self.reduced_covariance


        return msfe, alpl, coeff, sigma, complete_msfe, complete_alpl

