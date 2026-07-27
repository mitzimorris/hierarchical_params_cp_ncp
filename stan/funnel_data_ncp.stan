data {
  int<lower=0> N;                    // count trials
  array[9] int<lower=0, upper=N> y;  // count successes
}
parameters {
  real<lower=0> sigma;
  vector[9] theta_raw;
}
transformed parameters {
  vector[9] theta;                     // log odds success
  theta = sigma * theta_raw;
}
model {
  sigma ~ normal(0, 3);
  theta_raw ~ std_normal();
  y ~ binomial_logit(N, theta);
}
