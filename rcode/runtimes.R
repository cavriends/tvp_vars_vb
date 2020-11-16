setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# install.packages('vars')
# install.packages('BVAR')
# install.packages("MARX")
# install.packages('shrinkTVP')
#install.packages("coda")

library(vars) 
library(BVAR)
library(MARX)
library(shrinkTVP)
library(parallel)
library(mvtnorm)
library(coda)

#### FED DATA ####

# fred_ds <- read.csv("../data/normal_7_fred.csv")
# complete_y <- fred_ds[, c(2, 3, 4)]#, 5, 6, 7, 8)]
# transformed_y <- fred_transform(complete_y, codes = c(5, 6, 2), scale=1) #, 6, 6, 5, 5), scale=1)

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

#### ARX with OLS ####

arx_ols <- function(T, M, p, train, y, x, h_steps) {
  
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
  
  
  metrics_tvp_ols <- h_step_metrics(h_steps, prediction_list, y, train)
  
  return(metrics_tvp_ols)
  
}

#### ARX with OLS (MSD) ####

arx_ols_msd <- function(T, M, p, y, x, true_coeff) {
  
  estimated_coeff <- c()
  
  for (m in 1:M) {
    
    y_all <- as.matrix(y[,m])
    x_all <- as.matrix(x)
    
    if (m != 1) {
      contemporaneous_y <- as.matrix(y[,seq(m-1)])
      x_all_complete <- cbind(x_all, contemporaneous_y)
      
    } else {
      
      x_all_complete <- x_all
      
    }
    
    arx_fit = arx.ls(y_all, x_all_complete, p=0)
    estimated_coeff <- cbind(estimated_coeff, arx_fit$coefficients[2:(M+1)])
  }
  
  msd <- c()
  
  for (i in 1:(T-1)) {
    msd <- rbind(msd, mean(estimated_coeff - matrix(true_coeff[,i], nrow=M, ncol=M))^2)
  }
  
  final_msd <- mean(msd)
  
  return(final_msd)
  
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
                           sv=FALSE,
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
  
  metrics_tvp_bar <- h_step_metrics(h_steps, prediction_list_tvp, y, train)
  
  return(metrics_tvp_bar)
  
}

#### TVP-B-AR with Minnesota prior (MSD) ####

tvp_bar_msd <- function(T, M, p, y, x, mcmc_iter, true_coeff, print_status=FALSE) {
  
  estimated_coeff <- c()
  
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
    
    estimated_coeff <- cbind(estimated_coeff, matrix(unlist(lapply(tvp_fit$beta, colMeans)), T)[,2:(M+1)])
    
  }
  
  final_msd <- mean(colMeans((t(estimated_coeff)[,2:T] - true_coeff)^2))
  
  return(final_msd)
  
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

#### SIMULATION (it is embarrassingly parallel)  ####

simulation_run <- function(run, M, mcmc_iter, sparsity=0.40) {
  
  T = 100
  train <- T+1-25
  number_of_predictions <- T - train
  h_steps <- 8
  p <- 1
  
  y_dgp <- read.csv(paste("../simulations/datasets/",paste('y', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  x_dgp <- read.csv(paste("../simulations/datasets/",paste('x', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  
  # Minimal error handling is necessary due to possible initialization failures for priors in the Bayesian models. OLS based models shouldn't error out.
  # However, as a precaution and in the name of consistency, error handling is also set up for these models.
  
  # var_result <- tryCatch({
  #   ## VAR-OLS ##
  #   start_time = proc.time()
  #   msfe_var_ols <- var_ols(T, M, p, train, y_dgp, h_steps)
  #   msd_var_ols <- var_ols_msd(T, M, p, y_dgp, coeff)
  #   elapsed_time = round((proc.time() - start_time)[3],4)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","VAR -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_var_ols$msfe)),4)," | ALPL: ", round(mean(unlist(msfe_var_ols$alpl))), " | MSD: ", round(msd_var_ols,6), "\n", sep=""))
  #   
  #   var_result = list("msfe_var_ols" = msfe_var_ols,
  #                     "msd_var_ols" = msd_var_ols)
  #   
  # }, error = function(err) {
  #   var_result = invalid_result(model="var_ols", h_steps)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","VAR -> ERROR!" ))
  # })
  # 
  # bvar_result = tryCatch({
  #   ## B-VAR with Minnesota prior ##
  #   start_time = proc.time()
  #   msfe_bvar_minnesota <- bvar_minnesota(T, M, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
  #   msd_bvar_minnesota <- bvar_minnesota_msd(T, M, p, y_dgp, mcmc_iter, coeff)
  #   elapsed_time = round((proc.time() - start_time)[3],4)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","BVAR -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_bvar_minnesota$msfe)),4), " | ALPL: ", round(mean(unlist(msfe_bvar_minnesota$alpl)))," | MSD: ", round(msd_bvar_minnesota,6), "\n", sep=""))
  # 
  #   bvar_result = list("msfe_bvar_minnesota" = msfe_bvar_minnesota,
  #                      "msd_bvar_minnesota" = msd_bvar_minnesota)
  # }, error = function(err) {
  #   bvar_result = invalid_result(model="bvar_minnesota", h_steps)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","BVAR -> ERROR!"))
  # })

  # arx_result = tryCatch({
  #   ## ARX with OLS ##
  #   start_time = proc.time()
  #   msfe_arx_ols <- arx_ols(T, M, p, train, y_dgp, x_dgp, h_steps)
  #   msd_arx_ols <- arx_ols_msd(T, M, p, y_dgp, x_dgp, coeff)
  #   elapsed_time = round((proc.time() - start_time)[3],4)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","AR-X -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_arx_ols$msfe)),4)," | ALPL: ", round(mean(unlist(msfe_arx_ols$alpl))), " | MSD: ", round(msd_arx_ols,6), "\n", sep=""))
  #   
  #   arx_result = list("msfe_arx_ols" = msfe_arx_ols,
  #                     "msd_arx_ols" = msd_arx_ols)
  # }, error = function(err) {
  #   arx_result = invalid_result(model="arx_ols", h_steps)
  #   cat(paste("Run: ", run, "\t", "M: ", M, "\t","AR-X -> ERROR!"))
  # })

  tvp_bar_result = tryCatch({
    ## TVP-B-AR with Minnesota prior ##
    start_time = proc.time()
    msfe_tvp_bar <- tvp_bar(T, M, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
    msd_tvp_bar <- tvp_bar_msd(T, M, p, y_dgp, x_dgp, mcmc_iter, coeff)
    elapsed_time = round((proc.time() - start_time)[3],4)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","TVP-B-AR -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_tvp_bar$msfe)),4)," | ALPL: ", round(mean(unlist(msfe_tvp_bar$alpl))), " | MSD: ", round(msd_tvp_bar,6), "\n", sep=""))

    tvp_bar_result = list("msfe_tvp_bar" = msfe_tvp_bar,
                          "msd_tvp_bar" = msd_tvp_bar)
  }, error = function(err) {
    tvp_bar_result = invalid_result(model="tvp_bar", h_steps)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","TVP-B-AR -> ERROR!"))
  })
  
  # result_list <- list("msfe_var_ols" = var_result$msfe_var_ols$msfe,
  #                     "alpl_var_ols" = var_result$msfe_var_ols$alpl,
  #                     "msfe_bvar_minnesota" = bvar_result$msfe_bvar_minnesota$msfe,
  #                     "alpl_bvar_minnesota" = bvar_result$msfe_bvar_minnesota$alpl,
  #                     "msfe_arx_ols" = arx_result$msfe_arx_ols$msfe,
  #                     "alpl_arx_ols" = arx_result$msfe_arx_ols$alpl,
  #                     "msfe_tvp_bar" = tvp_bar_result$msfe_tvp_bar$msfe,
  #                     "alpl_tvp_bar" = tvp_bar_result$result$msfe_tvp_bar$alpl,
  #                     "msd_var_ols" = var_result$msd_var_ols,
  #                     "msd_bvar_minnesota" = bvar_result$msd_bvar_minnesota,
  #                     "msd_arx_ols" = arx_result$msd_arx_ols,
  #                     "msd_tvp_bar" = tvp_bar_result$msd_tvp_bar)
  
  return(0)
  
}

cl_args <- commandArgs(trailingOnly = TRUE)
set.seed(12345) # For reproducability
m_list <- c(7)
mcmc_iter_list <- c(2000,1500,500)
n_iterations <- 16 #Number of cores on an Intel i9-9880H
iterations <- seq(n_iterations)
p <- 1
T = 100
sparsity = 0.40

for (m in m_list) {
  start_time = proc.time()
  results <- mclapply(iterations, simulation_run, M = m, mcmc_iter = mcmc_iter_list[which(m_list == m)], mc.cores = detectCores())
  file_string <- paste(paste("statistics", m, n_iterations, p, T, sparsity, "R", sep="_"), ".RData", sep="")
  # save(results, file=file_string)
  elapsed_time = round((proc.time() - start_time)[3],4)
  cat("\n")
  cat(paste("Average runtime for TVP-BVAR: ", elapsed_time, sep=""))
}
