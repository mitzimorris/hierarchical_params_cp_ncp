data {
  int<lower=0> N;                    // count trials
  array[9] int<lower=0, upper=N> y;  // count successes
}
parameters {
  real<lower=0> sigma;
  vector[9] theta;                  // log odds success
}
model {
  sigma ~ normal(0, 3);
  theta ~ normal(0, sigma);
  y ~ binomial_logit(N, theta);
}
