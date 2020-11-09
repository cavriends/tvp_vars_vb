setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

load("../simulations/results/statistics_7_200_1_200_0.4_R.RData")

n_iterations = length(results)

msfe_var_ols = c()
msfe_bvar = c()
msfe_tvp_bar = c()
msd_var_ols = c()
msd_bvar = c()
msd_tvp_bar = c()

for (iteration_result in results) {
  
  msfe_var_ols = rbind(msfe_var_ols, unlist(iteration_result$msfe_var_ols))
  msfe_bvar = rbind(msfe_bvar, unlist(iteration_result$msfe_bvar))
  msfe_tvp_bar = rbind(msfe_tvp_bar, unlist(iteration_result$msfe_tvp_bar))
  msd_var_ols = rbind(msd_var_ols, unlist(iteration_result$msd_var_ols))
  msd_bvar= rbind(msd_bvar, unlist(iteration_result$msd_bvar_minnesota))
  msd_tvp_bar = rbind(msd_tvp_bar, unlist(iteration_result$msd_tvp_bar))
  
}

threshold_percentage = 2.5e-2
threshold_high = round(n_iterations - threshold_percentage*n_iterations)
threshold_low = round(n_iterations*threshold_percentage)

#### VAR-OLS ####

mean_var_ols = rowMeans(msfe_var_ols)
indices_var_ols = sort(mean_var_ols, index.return = TRUE)$ix
cleaned_mean_var_ols = rowMeans(msfe_var_ols[indices_var_ols[threshold_low:threshold_high],])
msfe_h_step_var_ols = colMeans(msfe_var_ols[indices_var_ols[threshold_low:threshold_high],])
overall_msfe_var_ols = mean(cleaned_mean_var_ols)

#### B-VAR with Minnesota prior ####

mean_bvar = rowMeans(msfe_bvar)
indices_bvar = sort(mean_bvar, index.return = TRUE)$ix
cleaned_mean_bvar = rowMeans(msfe_bvar[indices_bvar[threshold_low:threshold_high],])
msfe_h_step_bvar = colMeans(msfe_bvar[indices_bvar[threshold_low:threshold_high],])
overall_msfe_bvar = mean(cleaned_mean_bvar)

#### TVP-B-AR with Minnesota prior ####

mean_tvp = rowMeans(msfe_tvp_bar)
indices_tvp = sort(mean_tvp, index.return = TRUE)$ix
cleaned_mean_tvp = rowMeans(msfe_tvp_bar[indices_tvp[threshold_low:threshold_high],])
msfe_h_step_tvp = colMeans(msfe_tvp_bar[indices_tvp[threshold_low:threshold_high],])
overall_msfe_tvp = mean(cleaned_mean_tvp)
