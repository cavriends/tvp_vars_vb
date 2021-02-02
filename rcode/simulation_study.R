setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# The simulation_study.R script carries out the simulation study. There are several important parameters.
# mcmc_iter_list: is a list that contains the number of MCMC iterations for the BVAR and TVP-BVAR
# n_iterations: the number of simulation iterations, in the thesis this is constrained at 200
# m_list: is a list of the different Ms, in the thesis these are 3 & 7
# T_list: is a list of the different time horizons, in the thesis these are 100 & 200
# sparsity_list: is a list containing the different percentages of sparsity, default are 0.20 (80%) and 0.40 (60%)
# p: is the number of lags, default in the thesis is 1
# An extra note, it is completely parallized code using mclapply(). This does not print to the Rstudio command line.
# Therefore, if one would like to monitor progress, one has to run it with Rscript in a terminal.

source("functions.R")

simulation_run <- function(run, M, mcmc_iter, T, sparsity) {
  
  train <- T+1-25
  number_of_predictions <- T - train
  h_steps <- 8
  p <- 1
  
  y_dgp <- read.csv(paste("../simulations/datasets/",paste('y', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  x_dgp <- read.csv(paste("../simulations/datasets/",paste('x', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
  
  # Minimal error handling is necessary due to possible initialization failures for priors in the Bayesian models. OLS based models shouldn't error out.
  # However, as a precaution and in the name of consistency, error handling is also set up for these models.
  
  var_result <- tryCatch({
    ## VAR-OLS ##
    start_time = proc.time()
    msfe_var_ols <- var_ols(T, M, p, train, y_dgp, h_steps)
    msd_var_ols <- var_ols_msd(T, M, p, y_dgp, coeff)
    elapsed_time = round((proc.time() - start_time)[3],4)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","VAR -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_var_ols$msfe)),4)," | ALPL: ", round(mean(unlist(msfe_var_ols$alpl))), " | MSD: ", round(msd_var_ols,6), "\n", sep=""))

    var_result = list("msfe_var_ols" = msfe_var_ols,
                      "msd_var_ols" = msd_var_ols)

  }, error = function(err) {
    var_result = invalid_result(model="var_ols", h_steps)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","VAR -> ERROR!" ))
  })
  
  bvar_result = tryCatch({
    ## B-VAR with Minnesota prior ##
    start_time = proc.time()
    msfe_bvar_minnesota <- bvar_minnesota(T, M, p, train, y_dgp, x_dgp, mcmc_iter, h_steps)
    msd_bvar_minnesota <- bvar_minnesota_msd(T, M, p, y_dgp, mcmc_iter, coeff)
    elapsed_time = round((proc.time() - start_time)[3],4)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","BVAR -> DONE! | elapsed: ", elapsed_time, ' seconds | MSFE: ',  round(mean(unlist(msfe_bvar_minnesota$msfe)),4), " | ALPL: ", round(mean(unlist(msfe_bvar_minnesota$alpl)))," | MSD: ", round(msd_bvar_minnesota,6), "\n", sep=""))

    bvar_result = list("msfe_bvar_minnesota" = msfe_bvar_minnesota,
                       "msd_bvar_minnesota" = msd_bvar_minnesota)
  }, error = function(err) {
    bvar_result = invalid_result(model="bvar_minnesota", h_steps)
    cat(paste("Run: ", run, "\t", "M: ", M, "\t","BVAR -> ERROR!"))
  })
  
  
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
  
  
  result_list <- list("msfe_var_ols" = var_result$msfe_var_ols$msfe,
                      "alpl_var_ols" = var_result$msfe_var_ols$alpl,
                      "msfe_bvar_minnesota" = bvar_result$msfe_bvar_minnesota$msfe,
                      "alpl_bvar_minnesota" = bvar_result$msfe_bvar_minnesota$alpl,
                      "msfe_tvp_bar" = tvp_bar_result$msfe_tvp_bar$msfe,
                      "alpl_tvp_bar" = tvp_bar_result$msfe_tvp_bar$alpl,
                      "msd_var_ols" = var_result$msd_var_ols,
                      "msd_bvar_minnesota" = bvar_result$msd_bvar_minnesota,
                      "msd_tvp_bar" = tvp_bar_result$msd_tvp_bar)
  
  return(result_list)

}

cl_args <- commandArgs(trailingOnly = TRUE)
set.seed(12345) # For reproducability
mcmc_iter_list <- c(2000,1000)
n_iterations <- as.numeric(cl_args[1]) # Default in paper is 200 
iterations <- seq(n_iterations)
m_list <- c(3,7)
T_list = c(100, 200)
sparsity_list = c(0.20, 0.40)
p <- 1

for (m in m_list) {
  for (sparsity in sparsity_list){
    for (T in T_list) {
      results <- mclapply(iterations, simulation_run, M = m, mcmc_iter = mcmc_iter_list[which(m_list == m)], T=T, sparsity=sparsity, mc.cores = detectCores())
      file_string <- paste(paste("statistics", m, n_iterations, p, T, sparsity, "R", "huber", sep="_"), ".RData", sep="")
      save(results, file=file_string) 
      
    }
  }
}