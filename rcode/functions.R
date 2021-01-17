setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# install.packages('vars')
# install.packages('BVAR')
# install.packages("MARX")
# install.packages('shrinkTVP')
# install.packages('stableGR')

library(vars) 
library(BVAR)
library(MARX)
library(shrinkTVP)
library(parallel)
library(mvtnorm)
library(coda)

standardize <- function(data, train) {
  
  standardized = c()
  
  for (m in 1:dim(data)[2]) {
    
    x = data[,m]
    
    mean = mean(x[1:train])
    std = sd(x[1:train])
    
    standardized = cbind(standardized, (x - mean)/std)
    
  }
  
  return(standardized)
  
}

h_step_metrics <- function(h_step, y_pred, y_true, train) {
  
  msfe_list = list()
  alpl_list = list()
  
  for (h in 0:(h_step-1)) {
    
    msfe_vec <- c()
    alpl_vec <- c()
    y_pred_h <- c()
    y_true_h <- c()
    
    number_of_predictions = dim(y_true)[1] - train - h
    
    for (i in 1:number_of_predictions) {
      
      msfe = (y_pred[[i]][h+1,] - y_true[train+i+h,])^2
      msfe_vec = rbind(msfe_vec, msfe)
      
      y_pred_h <- rbind(y_pred_h, y_pred[[i]][h+1,] ) 
      y_true_h <- rbind(y_true_h, y_true[train+i+h,])
      
    }
    
    for (i in 1:number_of_predictions) {
      
      alpl = log(dmvnorm(y_true_h[i,], y_pred_h[i,], cov(y_pred_h), checkSymmetry = FALSE) + 1e-16)
      alpl_vec = rbind(alpl_vec, alpl)
      
    }
    
    msfe_list = append(msfe_list, mean(colMeans(msfe_vec)))
    alpl_list = append(alpl_list, mean(alpl))
  }
  
  metrics_list <- list("msfe" = msfe_list, 
                       "alpl" = alpl_list)
  
  return(metrics_list)
  
}

#### VAR-OLS ####

var_ols <- function(T, M, p, train, y, h_steps) {
  
  predict_list_var <- list()
  
  for (i in train:(T - 1)) {
    y_subset <- y[1:i, ]
    var_fit <- VAR(y_subset, lag = p)
    predict_result <- predict(var_fit, n.ahead = h_steps)$fcst
    
    predictions <- c()
    
    for (m in 1:M) {
      predictions <- cbind(predictions, predict_result[[m]][, 1])
    }
    
    predict_list_var <- append(predict_list_var, list(predictions))
  }
  
  metrics_var <- h_step_metrics(h_steps, predict_list_var, y, train)
  
  return(metrics_var)
  
}

## VAR-OLS (MSD) ##

var_ols_msd <- function(T, M, p, y, true_coeff) {
  
  var_fit <- VAR(y, lag = p)
  
  estimated_coeff <- c()
  for (m in 1:M) {
    estimated_coeff <- cbind(estimated_coeff, var_fit$varresult[[m]][[1]][1:M])
  }
  
  msd <- c()
  
  for (i in 1:(T-1)) {
    msd <- rbind(msd, mean(estimated_coeff - matrix(true_coeff[,i], nrow=M, ncol=M))^2)
  }
  
  final_msd <- mean(msd)
  
  return(final_msd)
  
}

#### B-VAR with Minnesota prior ####

bvar_minnesota <- function(T, M, p, train, y, x, mcmc_iter, h_steps, print_status=FALSE) {
  
  predict_list_bvar <- list()
  mcmc_iter_bvar = mcmc_iter*5
  
  for (i in train:(T - 1)) {
    
    if (print_status) {
      
      if (i %% 10 == 0) {
        cat("\014")
        print("Progress of B-VAR with Minnesota prior...")
      }
      
      print(sprintf("Progress %i/%i", i - train + 1, number_of_predictions))
      
    }
    
    y_subset <- y[1:i, ]
    bvar_fit <- bvar(data = y_subset,
                     lags = p,
                     #priors = bv_priors(hyper = c("full")),
                     fcast = bv_fcast(h_steps),
                     verbose = FALSE,
                     n_draw = mcmc_iter_bvar,
                     n_burn = floor(mcmc_iter_bvar/2))
    
    predict_result <- predict(bvar_fit)
    
    predictions <- c()
    
    for (h in 1:h_steps) {
      predictions <- rbind(predictions, colMeans(predict_result$fcast[, h, ]))
    }
    
    predict_list_bvar <- append(predict_list_bvar, list(predictions))
    
  }
  
  metrics_bvar <- h_step_metrics(h_steps, predict_list_bvar, y, train)
  
  return(metrics_bvar)
  
}

#### B-VAR with Minnesota prior (MSD) ####

bvar_minnesota_msd <- function(T, M, p, y, mcmc_iter, true_coeff, print_status=FALSE) {
  
  mcmc_iter_bvar = mcmc_iter*5
  
  bvar_fit <- bvar(data = y,
                   lags = p,
                   #priors = bv_priors(hyper = c("full")),
                   verbose = FALSE,
                   n_draw = mcmc_iter_bvar,
                   n_burn = floor(mcmc_iter_bvar/2))
  
  
  estimated_coeff <- colMeans(bvar_fit$beta)[2:(M+1),]
  
  msd <- c()
  
  for (i in 1:(T-1)) {
    msd <- rbind(msd, mean(estimated_coeff - matrix(true_coeff[,i], nrow=M, ncol=M))^2)
  }
  
  final_msd <- mean(msd)
  
  return(final_msd)
}

#### B-VAR with Minnesota prior (GR diagnostic) ####

bvar_minnesota_GR <- function(T, M, p, y, mcmc_iter, mc_chains=4, print_status=FALSE) {
  
  mcmc_list <- list()
  
  for (i in 1:mc_chains) {
    
    mcmc_iter_bvar = mcmc_iter*5
    
    bvar_fit <- bvar(data = y,
                     lags = p,
                     verbose = FALSE,
                     n_draw = mcmc_iter_bvar,
                     n_burn = floor(mcmc_iter_bvar/2))
    
    mcmc_list <- append(mcmc_list, list(as.mcmc(matrix(bvar_fit$beta, nrow=floor(mcmc_iter_bvar/2), ncol=M*(M+1)))))
    
  }
  
  mcmc_list <- as.mcmc.list(mcmc_list)
  mpsrf <- gelman.diag(mcmc_list)$mpsrf
  
  return(mpsrf)
  
}

#### TVP-B-AR with Minnesota prior ####

tvp_bar <- function(T, M, p, train, y, x, mcmc_iter, h_steps, print_status=FALSE) {
  
  prediction_list_tvp <- list()
  
  for (i in train:(T - 1)) {
    
    if (print_status) {
      
      if (i %% 10 == 0) {
        cat("\014")
        print("Progress of TVP-B-AR with Minnesota prior...")
      }
      
      print(sprintf("Progress %i/%i", i - train + 1, number_of_predictions))
      
    }   
    
    prediction <- c()
    structural_coefficients <- matrix(rep(0, M*(M*p+1)), nrow=M, ncol=M*p+1)
    reduced_coefficients <- matrix(rep(0, M*(M*p+1)), nrow=M, ncol=M*p+1)
    U_elements <- c()
    lower_tri_U <- diag(M)
    
    for (m in 1:M) {
      
      y_subset <- y[1:i,m]
      x_subset <- x[1:i,]
      
      if (m != 1) {
        contemporaneous_y <- y[1:i,seq(m-1)]
        x_complete <- cbind(x_subset, contemporaneous_y)
        
      } else {
        
        x_complete <- x_subset
        
      }
      
      variable_string <- c()
      number_of_variables <- dim(x_complete)[2]
      
      for (v in 1:number_of_variables) {
        variable_string <- c(variable_string, paste("V", v, sep=""))
      }
      
      colnames(x_complete) <- variable_string
      
      data <- cbind(y_subset,x_complete)
      
      tvp_fit <- shrinkTVP(y_subset ~ ., 
                           data=data, 
                           niter=mcmc_iter, 
                           nburn=floor(mcmc_iter/2),
                           sv=TRUE,
                           display_progress=FALSE)
      
      beta_mean <- as.numeric(unlist(lapply(lapply(tvp_fit$beta, colMeans), tail, 1)))
      structural_coefficients[m,] = beta_mean[0:M+1]
      
      if (m != 1) {
        U_elements = append(U_elements, tail(beta_mean, -(M+1)))
      }
      
    }
    
    lower_tri_U[lower.tri(lower_tri_U)] = -U_elements
    lower_tri_U_inv <- solve(lower_tri_U)
    reduced_coefficients = lower_tri_U_inv %*% structural_coefficients
    
    h_prediction <- c()
    
    for (h in 1:h_steps) {
      
      if (h == 1) {
        h_prediction <- cbind(h_prediction, reduced_coefficients %*% t(as.matrix(cbind(1, x[i,]))))
      } 
      
      else {
        
        h_prediction <- cbind(h_prediction, reduced_coefficients %*% as.matrix(append(1,h_prediction[,h-1])))
        
      }
      
    }
    
    prediction <- h_prediction
    
    prediction_list_tvp <- append(prediction_list_tvp, list(t(prediction)))
    
  }
  
  metrics_tvp_bar <- h_step_metrics(h_steps, prediction_list_tvp, y, train)
  
  return(metrics_tvp_bar)
  
}

#### TVP-B-AR with Minnesota prior (MSD) ####

tvp_bar_msd <- function(T, M, p, y, x, mcmc_iter, true_coeff, print_status=FALSE) {
  
  structural_coeff <- list()
  elements_of_U <- vector('list', T)
  reduced_form_coeff <- vector('list', T)
  
  for (i in 1:T) {
    structural_coeff = append(structural_coeff, list(matrix(rep(0, M*(M*p+1)), nrow=M, ncol=M*p+1)))
  }
  
  for (m in 1:M) {
    
    y_all <- y[,m]
    x_all <- x
    
    if (m != 1) {
      contemporaneous_y <- y[,seq(m-1)]
      x_all_complete <- cbind(x_all, contemporaneous_y)
      
    } else {
      
      x_all_complete <- x_all
      
    }
    
    variable_string <- c()
    number_of_variables <- dim(x_all_complete)[2]
    
    for (v in 1:number_of_variables) {
      variable_string <- c(variable_string, paste("V", v, sep=""))
    }
    
    colnames(x_all_complete) <- variable_string
    
    data <- cbind(y_all,x_all_complete)
    
    tvp_fit <- shrinkTVP(y_all ~ ., 
                         data=data, 
                         niter=mcmc_iter, 
                         nburn=floor(mcmc_iter/2), 
                         display_progress=FALSE)
    
    
    
    estimated_coeff <-  matrix(unlist(lapply(tvp_fit$beta, colMeans)), T)
    
    for (i in 1:T) {
      structural_coeff[[i]][m,] =  estimated_coeff[i,0:M+1]
      if (m != 1) {
        if (m == 2) {
          elements_of_U[[i]] = tail(estimated_coeff[i,], -(M+1))
        } else {
          elements_of_U[[i]] = append(elements_of_U[[i]], tail(estimated_coeff[i,], -(M+1))) 
        }
      }
    }
  }
  
  for (i in 1:T) {
    U <- diag(M)
    U[lower.tri(U)] = -elements_of_U[[i]]
    reduced_form_coeff[[i]] = as.vector((solve(U) %*% structural_coeff[[i]])[,1:M+1])
  }
  
  final_msd <- mean(colMeans((matrix(unlist(reduced_form_coeff), M*M)[,2:T] - true_coeff)^2))
  
  return(final_msd)
  
}

#### TVP-B-AR with Minnesota prior (GR diagnostic)

tvp_bar_GR <- function(T, M, p, y, x, mcmc_iter=1000, mc_chains=4) {
  
  chain_output = list()
  
  for (i in 1:mc_chains) {
    
    mcmc_output = list()
    
    structural_coeff <- list()
    elements_of_U <- vector('list', T)
    reduced_form_coeff <- vector('list', T)
    
    for (i in 1:T) {
      structural_coeff = append(structural_coeff, list(matrix(rep(0, M*(M*p+1)), nrow=M, ncol=M*p+1)))
    }
    
    for (m in 1:M) {
      
      y_all <- y[,m]
      x_all <- x
      
      if (m != 1) {
        contemporaneous_y <- y[,seq(m-1)]
        x_all_complete <- cbind(x_all, contemporaneous_y)
        
      } else {
        
        x_all_complete <- x_all
        
      }
      
      variable_string <- c()
      number_of_variables <- dim(x_all_complete)[2]
      
      for (v in 1:number_of_variables) {
        variable_string <- c(variable_string, paste("V", v, sep=""))
      }
      
      colnames(x_all_complete) <- variable_string
      
      data <- cbind(y_all,x_all_complete)
      
      tvp_fit <- shrinkTVP(y_all ~ ., 
                           data=data, 
                           niter=mcmc_iter, 
                           nburn=floor(mcmc_iter/2), 
                           display_progress=FALSE)
      
      mcmc_output = append(mcmc_output, list(tvp_fit$beta))
      
    }
    
    chain_output = append(chain_output, list(mcmc_output))
    
  }
  
  # Calculate the multivariate PSRF (Rubin-Gelman statistic)
  
  mpsrf_list <- c()
  
  for (i in 1:M){
    
    for (m in 1:(M+i)) {
      
      gr_list <- list()
      mcmc_list <- list()
      
      for (chain in 1:mc_chains) {
        
        mcmc_list <- append(mcmc_list, list(as.mcmc(matrix(chain_output[[chain]][[i]][[m]], mcmc_iter/2, T))))
        
      }
      
      mcmc_list <- as.mcmc.list(mcmc_list)
      mpsrf_list <- cbind(mpsrf_list, gelman.diag(mcmc_list)$mpsrf)
      
    }
    
  }
  
  return(mean(mpsrf_list))
  
}

#### ERROR HANDLING ####

invalid_result <- function(model, h_steps=8) {
  
  invalid_list = list()
  
  for (i in 1:h_steps) {
    invalid_list = append(invalid_list, list(1e+4))
  }
  
  metrics_list = list("msfe" = invalid_list,
                      "alpl" = invalid_list)
  
  msfe_string = paste("msfe_", model, sep="")
  msd_string = paste("msd_", model, sep="")
  
  return_list = list(metrics_list,
                     1e+4)
  
  names(return_list) <- c(msfe_string, msd_string)
  
  return(return_list)
  
}