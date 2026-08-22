data {
  int<lower=0> N;                    // count trials
  array[9] int<lower=0, upper=N> y;  // count successes
}
parameters {
  real log_sigma_sq;
  vector[9] theta_std;
}
transformed parameters {
  vector[9] theta;                     // log odds success
  theta = exp(log_sigma_sq/2) * theta_std;
}
model {
  log_sigma_sq ~ normal(0, 3);
  theta_std ~ std_normal();
  y ~ binomial_logit(N, theta);
}
