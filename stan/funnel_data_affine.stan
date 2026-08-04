data {
  int<lower=0> N;                     // count trials
  array[9] int<lower=0, upper=N> y;   // count successes
}
parameters {
  real log_sigma_sq;
  vector<multiplier=exp(log_sigma_sq/2)>[9] theta;  // log odds success
}
model {
  sigma ~ normal(0, 3);
  theta ~ normal(0, exp(log_sigma_sq/2));
  y ~ binomial_logit(N, theta);
}
