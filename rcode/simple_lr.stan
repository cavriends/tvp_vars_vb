data {
  int<lower=0> N;
  vector[N] x_1;
  vector[N] x_2;
  vector[N] y;
}
parameters {
  real alpha;
  real beta_1;
  real beta_2;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta_1 * x_1 + beta_2 * x_2, sigma);
}