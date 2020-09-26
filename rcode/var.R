
setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# install.packages('vars')
# install.packages('BVAR')
# install.packages("MARX")
# install.packages('shrinkTVP')

library(vars)
library(BVAR)
library(MARX)
library(shrinkTVP)
library(parallel)

#fred_ds <- read.csv("../data/normal_7_fred.csv")
#complete_y <- fred_ds[, c(2, 3, 4)]#, 5, 6, 7, 8)]
#transformed_y <- fred_transform(complete_y, codes = c(5, 6, 2), scale=1) #, 6, 6, 5, 5), scale=1)

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

h_step_msfe <- function(h_step, y_pred, y_true, train) {
  
  msfe_list = list()
  
  for (h in 0:(h_step-1)) {
    
    msfe_vec = c()
    
    number_of_predictions = dim(y_true)[1] - train - h
    
    for (i in 1:number_of_predictions) {
      msfe = (y_pred[[i]][h+1,] - y_true[train+i+h,])^2
      msfe_vec = rbind(msfe_vec, msfe)
    }
    msfe_list = append(msfe_list, mean(colMeans(msfe_vec)))
  }
  
  return(msfe_list)
  
}

#### VAR-OLS ####

var_ols <- function(T, M, p, train, y, h_steps, coeff) {
  
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
  
  msfe_var <- h_step_msfe(h_steps, predict_list_var, y, train)
  
  
  
  return(msfe_var)
  
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
    msd <- rbind(msd, mean(estimated_coeff - matrix(true_coeff[,i], nrow=M, ncol=M)^2))
  }
  
  final_msd <- mean(msd)
  
  return(final_msd)
  
}

#### B-VAR with Minnesota prior ####

bvar_minnesota <- function(T, M, p, train, y, x, mcmc_iter, h_steps, coeff, print_status=FALSE) {
  
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
  
  msfe_bvar <- h_step_msfe(h_steps, predict_list_bvar, y, train)
  
  return(msfe_bvar)
  
}

#### ARX with OLS ####

arx_ols <- function(T, M, p, train, y, x, h_steps, coeff) {
  
  prediction_list <- list() 
  
  for (i in train:(T - 1)) {
    
    prediction <- c()  
    
    for (m in 1:M) {
      
      y_subset <- as.matrix(y[1:i,m])
      x_subset <- as.matrix(x[1:i,])
      
      if (m != 1) {
        contemporaneous_y <- as.matrix(y[1:i,seq(m-1)])
        x_complete <- cbind(x_subset, contemporaneous_y)
        
      } else {

        x_complete <- x_subset
        
      }
      
      arx_fit = arx.ls(y_subset, x_complete, p=0)
      
      h_prediction <- c()
      
      if (m != 1) {
        
        for (h in 1:h_steps) {
          h_prediction <- rbind(h_prediction, arx_fit$coefficients[2:(M+1+(m-1))] %*% t(cbind(as.matrix(x[i+h,]), as.matrix(y[i+h,seq(m-1)]))))
        } 
        
      } else {
        
        for (h in 1:h_steps) {
          h_prediction <- rbind(h_prediction, arx_fit$coefficients[2:(M+1)] %*% t(as.matrix(x[i+h,])))
        } 
        
      }
      
      prediction <- cbind(prediction, h_prediction)
      
    }
    
    prediction_list <- append(prediction_list, list(prediction))
  }
  
  
  msfe_tvp_ols <- h_step_msfe(h_steps, prediction_list, y, train)
  
  return(msfe_tvp_ols)
  
}

#### TVP-B-AR with Minnesota prior ####

tvp_bar <- function(T, M, p, train, y, x, mcmc_iter, h_steps, coeff, print_status=FALSE) {
 
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
    
    for (m in 1:M) {
      
      y_subset <- y[1:i,m]
      x_subset <- x[1:i,]
      
      if (m != 1) {
        contemporaneous_y <- y[1:i,seq(m-1)]
        x_complete <- cbind(x_subset, contemporaneous_y)
        
      } else {
        
        x_complete <- x_subset
        
      }
      
      data <- cbind(y_subset,x_complete)
      
      tvp_fit <- shrinkTVP(y_subset ~ ., 
                           data=data, 
                           niter=mcmc_iter, 
                           nburn=floor(mcmc_iter/2), 
                           display_progress=FALSE)
      
      beta_mean <- unlist(lapply(lapply(tvp_fit$beta, colMeans), tail, 1))
      
      h_prediction <- c()
      
      if (m != 1) {
        
        for (h in 1:h_steps) {
          h_prediction <- rbind(h_prediction, beta_mean[2:(M+1+(m-1))] %*% t(cbind(as.matrix(x[i+h,]), as.matrix(y[i+h,seq(m-1)]))))
        } 
        
      } else {
        
        for (h in 1:h_steps) {
          h_prediction <- rbind(h_prediction, beta_mean[2:(M+1)] %*% t(as.matrix(x[i+h,])))
        } 
        
      }
      
      # for (h in 1:h_steps) {
      #   h_prediction <- rbind(h_prediction, beta_mean[2:(M+1)] %*% t(as.matrix(x[i+h,])))  
      # }
      
      prediction <- cbind(prediction, h_prediction)
      
    }
    
    prediction_list_tvp <- append(prediction_list_tvp, list(prediction))
    
  }
  
  msfe_tvp_bar <- h_step_msfe(h_steps, prediction_list_tvp, y, train)
  
  return(msfe_tvp_bar)
  
}

####  Simulation ####

# n_iterations = 200
# M = c(2,5)
# T = 200
# train <- T+1-50
# number_of_predictions <- T - train
# h_steps <- 8
# p <- 1
# mcmc_iter <- 1000
# 
# msfe_list <- list()
# msd_list <- list()
# 
# for (m in M) {
#   
#   for (run in 1:n_iterations) {
#     
#     print(paste("M: ", m, " | iteration: ", run,"/", n_iterations, sep=""))
#     
#     y_dgp <- read.csv(paste("../simulations/datasets/",paste('y',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
#     x_dgp <- read.csv(paste("../simulations/datasets/",paste('x',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
#     coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
#     
#     ## VAR-OLS ##
#     start_time = proc.time()
#     msfe_var_ols <- var_ols(T, m, p, train, y_dgp, h_steps)
#     msd_var_ols <- var_ols_msd(T, m, p, y_dgp, coeff)
#     print(paste("VAR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_var_ols)),4), sep=""))
#     
#     ## B-VAR with Minnesota prior ##
#     start_time = proc.time()
#     msfe_bvar_minnesota <- bvar_minnesota(T, m, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
#     print(paste("BVAR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_bvar_minnesota)),4), sep=""))
#     
#     ## ARX with OLS ##
#     start_time = proc.time()
#     msfe_arx_ols <- arx_ols(T, m, p, train, y_dgp, x_dgp, h_steps)
#     print(paste("AR-X -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_arx_ols)),4), sep=""))
#     
#     ## TVP-B-AR with Minnesota prior ##
#     start_time = proc.time()
#     msfe_tvp_bar <- tvp_bar(T, m, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
#     print(paste("TVP-B-AR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_tvp_bar)),4), sep=""))
#     
#     msfe_list <- append(msfe_list, list(msfe_var_ols, msfe_bvar_minnesota, msfe_arx_ols, msfe_tvp_bar))
#     msd_list <- append(msd_list, list(msd_var_ols))
#     
#     if (run %% 10 == 0) {
#       cat("\014")
#     }
#   }
#   
# }

#### Parallelized ####

simulation_run <- function(run) {
  
  m = 5
  T = 200
  train <- T+1-50
  number_of_predictions <- T - train
  h_steps <- 8
  p <- 1
  mcmc_iter <- 1000
  
  y_dgp <- read.csv(paste("../simulations/datasets/",paste('y',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
  x_dgp <- read.csv(paste("../simulations/datasets/",paste('x',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
  coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
  
  ## VAR-OLS ##
  start_time = proc.time()
  msfe_var_ols <- var_ols(T, m, p, train, y_dgp, h_steps)
  msd_var_ols <- var_ols_msd(T, m, p, y_dgp, coeff)
  print(paste("VAR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_var_ols)),4), sep=""))
  
  ## B-VAR with Minnesota prior ##
  start_time = proc.time()
  msfe_bvar_minnesota <- bvar_minnesota(T, m, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
  print(paste("BVAR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_bvar_minnesota)),4), sep=""))
  
  ## ARX with OLS ##
  start_time = proc.time()
  msfe_arx_ols <- arx_ols(T, m, p, train, y_dgp, x_dgp, h_steps)
  print(paste("AR-X -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_arx_ols)),4), sep=""))
  
  ## TVP-B-AR with Minnesota prior ##
  start_time = proc.time()
  msfe_tvp_bar <- tvp_bar(T, m, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
  print(paste("TVP-B-AR -> DONE! | elapsed: ", round((proc.time() - start_time)[3],4), ' seconds | MSFE: ',  round(mean(unlist(msfe_tvp_bar)),4), sep=""))
  
  
  result_list <- list("msfe_var_ols" = msfe_var_ols,
                      "msfe_bvar_minnesota" = msfe_bvar_minnesota,
                      "msfe_arx_ols" = msfe_arx_ols,
                      "msfe_tvp_bar" = msfe_tvp_bar,
                      "msd_var_ols" = msd_var_ols)
  
  return(result_list)

}

n_iterations = 200
iterations <- seq(n_iterations)
results <- mclapply(iterations, simulation_run, mc.cores = detectCores())
save(results, file="statistics_5_200_1_R.RData")