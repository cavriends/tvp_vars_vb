functions {
  #include ssm.stan;
}

data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> K;
  int<lower=0> P;
  vector[N] y;
  matrix[M,N] X;
  vector[N] d;
  matrix[]
}

parameters {
  matrix[K,K] betas;
  corr_matrix[M] sigma_observation;
  corr_matrix[K] sigma_states;
  real<lower=0> sigma;
}

model {
  y ~ normal(mu, sigma);
  
  likelihood = ssm_lpdf(vector[] y,vector[] d, matrix[] Z, matrix[] H, vector[] c, matrix[] T, matrix[] R, matrix[] Q,vector a1, matrix P1) 
}