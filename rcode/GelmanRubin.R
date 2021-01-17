setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

source("functions.R")

set.seed(12345) # For reproducability

# Set the datasets
number_of_datasets = 20
random_datasets = sample(1:200, number_of_datasets)

# Set the dataset parameters
T_list = c(100, 200)
m_list <- c(3,7)
sparsity_list = c(0.20, 0.40)
mcmc_iter_list <- c(2000,1000)

gr_statistic_run <- function(T, M, mcmc_iter, sparsity, mc_chains=4) {
  
  p <- 1
  
  mean_diagnostic_tvp_bar = c()
  mean_diagnostic_bvar = c()
  
  for (run in random_datasets) {
    
    # Load simulated datasets
    y_dgp <- read.csv(paste("../simulations/datasets/",paste('y', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
    x_dgp <- read.csv(paste("../simulations/datasets/",paste('x', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
    coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients', M, T, p, sparsity, run, 'het',sep="_"),'.csv', sep=""), header=FALSE)
    
    # Calculate Gelman-Rubin diagnostic
    msrpf_tvp_bar <- tvp_bar_GR(T, M, p, y_dgp, x_dgp, mcmc_iter, mc_chains)
    msrpf_bvar <- bvar_minnesota_GR(T, M, p, y_dgp, mcmc_iter, mc_chains)
    
    # Add results to vector
    mean_diagnostic_tvp_bar <- rbind(mean_diagnostic_tvp_bar, msrpf_tvp_bar)
    mean_diagnostic_bvar <- rbind(mean_diagnostic_bvar, msrpf_bvar)
    
    # Print final result
    if (run == random_datasets[number_of_datasets]) {
      
      print(paste(M, T, sparsity, "average GR - TVP-BVAR", round(mean(mean_diagnostic_tvp_bar),4), "average GR - BVAR", round(mean(mean_diagnostic_bvar),4),sep=" | "))
      
    }
  }
  
  return_list <- list("mean_gr_tvp_bar" = mean(mean_diagnostic_tvp_bar),
                      "mean_gr_bvar" = mean(mean_diagnostic_bvar))
  
  return(return_list)
  
}

gr_result <- c()

for (m in m_list) {
  for (sparsity in sparsity_list){
    for (T in T_list) {
      gr_result <- cbind(gr_result, gr_statistic_run(T=T, M=m, mcmc_iter = mcmc_iter_list[which(m_list == m)], sparsity=sparsity))
    }
  }
}