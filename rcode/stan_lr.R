
setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

library(rstan)

lr_model <- "./simple_lr.stan"


# Simple linear model

true_beta_1 <- 3
true_beta_2 <- 7

x_1 <- rnorm(1000, 1)
x_2 <- rnorm(1000, 4)

y <- true_beta_1*x_1 + true_beta_2*x_2 + rnorm(1000) 

# OLS

ols_result <- lm(y ~ x_1 + x_2)
summary(ols_result)

# Using Stan

data <- within(list(), {
  y <- as.vector(y)
  x_1 <- as.vector(x_1)
  x_2 <- as.vector(x_2)
  N <- length(y)
})

# Hamiltonian MC
fit <- stan(lr_model, data=data)
summary(fit)

# Varational Inference
vi_model <- stan_model(lr_model)

beta_1 <- c()
beta_2 <- c()
alpha <- c()

for (i in 1:1000) { 
  
  tryCatch({
    
    vi_result <- vb(vi_model, data=data, iter=1000) 
    beta_1 <- cbind(beta_1, vi_result@sim$est$beta_1)
    beta_2 <- cbind(beta_2, vi_result@sim$est$beta_2)
    alpha <- cbind(alpha, vi_result@sim$est$alpha)  
  }, error=function(e){})
}