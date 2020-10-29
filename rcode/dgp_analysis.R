setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

#### Functions necessary for analysis ####

rename_results <- function(M) {
  file_name <- paste("results", M, sep="_")
  assign(file_name, get("results", envir=.GlobalEnv), envir=.GlobalEnv)
  rm("results", envir=.GlobalEnv)  
} 

get_metric_in_matrix <- function(result, metric_list, h_steps=8, n_iterations=200) {
  
  matrices <- list()
  
  for (model_metric in metric_list) {
    temporary_list <- list()
    for (i in 1:n_iterations) {
      temporary_list <- append(temporary_list, list(getElement(result[[i]], model_metric)))
    }
    
    if (substr(model_metric, 1, 3) == "msd") {
      matrices <- append(matrices, list(matrix(unlist(temporary_list), n_iterations)))
    } else {
      matrices <- append(matrices, list(t(matrix(unlist(temporary_list), h_steps))))
    }
    
  }
  
  result_list <- list(matrices)
  names(result_list) <- model_metric 
  
  return(result_list)
  
} 

#### Analysis of simulation results  ####

# Simulation parameters
n_iterations <- 200 
h_steps <- 8
M <- c(2,5,10)

# Load datasets
for (m in M) {
  file_name <- paste(paste("statistics", m, n_iterations, 1, "R", sep="_"), ".RData", sep="")
  load(file_name)
  rename_results(m)
}

# Get matrics for all results in matrices
msfe_list <- c('msfe_var_ols', 'msfe_bvar_minnesota', 'msfe_arx_ols', 'msfe_tvp_bar')
alpl_list <- c('alpl_var_ols', 'alpl_bvar_minnesota', 'alpl_arx_ols', 'alpl_tvp_bar')
msd_list <- c('msd_var_ols', 'msd_bvar_minnesota', 'msd_arx_ols', 'msd_tvp_bar')
stacked_metrics <- c(msfe_list, alpl_list, msd_list)

metrics_10 <- list()
metrics_5 <- list()
metrics_2 <- list()

for (metric_list in stacked_metrics) {
  metrics_10 <- append(metrics_10, get_metric_in_matrix(results_10, metric_list))
  metrics_5 <- append(metrics_5, get_metric_in_matrix(results_5, metric_list))
  metrics_2 <- append(metrics_2, get_metric_in_matrix(results_2, metric_list))
}

# for (msfe_results in msfe_matrices) {
#   temporary_msfe_list <- list()
#   temporary_std_list <- list()
#   for (i in 1:n_iterations) {
#     if (i == 1) {
#       temporary_msfe_list <- append(temporary_msfe_list, list(mean(msfe_results[1,])))
#       temporary_std_list <- append(temporary_std_list, list(0))
#     } else {
#       temporary_msfe_list <- append(temporary_msfe_list, list(mean(rowMeans(msfe_results[1:i,]))))
#       temporary_std_list <- append(temporary_std_list, list(sd(rowMeans(msfe_results[1:i,]))))
#     }
#   }
#   meaned_msfe_list <- append(meaned_msfe_list, list(temporary_msfe_list))
#   std_msfe_list <- append(std_msfe_list, list(temporary_std_list))
# }
# 
# plot(x=seq(n_iterations), y=meaned_msfe_list[[4]],  type="l", xlim = c(25,200))
# plot(x=seq(n_iterations), y=std_msfe_list[[1]],  type="l", xlim = c(10,200))