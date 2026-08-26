data {
  int<lower=0> N;                    // count trials
  array[9] int<lower=0, upper=N> y;  // count successes
}
parameters {
  real log_sigma_sq_std;
  vector[9] theta_std;
}
transformed parameters {
  real log_sigma_sq = 3.0 * log_sigma_sq_std;
  vector[9] theta = exp(log_sigma_sq/2) * theta_std;
}
model {
  log_sigma_sq_std ~ std_normal();
  theta_std ~ std_normal();
  y ~ binomial_logit(N, theta);
}
