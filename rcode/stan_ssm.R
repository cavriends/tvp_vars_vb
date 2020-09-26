setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# install.packages("Rcpp", repos = "https://rcppcore.github.io/drat")
# 
# remove.packages("rstan")
# if (file.exists(".RData")) file.remove(".RData")
# 
# Sys.setenv(MAKEFLAGS = "-j8") # eight cores used
# 
# install.packages("rstan", type = "source")

var_model <- stanc_builder("var.stan", isystem = "./")
stan(model_code = model$model_code)


# Load a sample dataset from thesis
m = 2
T = 200
p = 1
run = 1

y_dgp <- read.csv(paste("../simulations/datasets/",paste('y',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
x_dgp <- read.csv(paste("../simulations/datasets/",paste('x',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)
coeff <- read.csv(paste("../simulations/datasets/",paste('coefficients',m,T,p,run,sep="_"),'.csv', sep=""), header=FALSE)

