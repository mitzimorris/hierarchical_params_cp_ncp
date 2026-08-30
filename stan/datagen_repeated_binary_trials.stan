// generate raw data for largest possible dataset
data {
  int<lower=0> N_groups;
  int<lower=0> N_obs;
}
generated quantities {
  vector[N_groups] theta;
  for (i in 1:N_groups) {
    theta[i] = std_normal_rng();  // mu = 0, sigma = 1
  }
  vector[N_groups] p = inv_logit(theta);  // probability in range (0,1)
  array[N_obs, N_groups] int<lower=0, upper=1> y;
  for (i in 1:N_obs) {
    for (j in 1:N_groups) {
      y[i, j] = bernoulli_rng(p[j]);
    }
  }
}
