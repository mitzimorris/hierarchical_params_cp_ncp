parameters {
  real sigma_std;
  vector[9] theta_std;
}
transformed parameters {
  real sigma;
  vector[9] theta;

  sigma = 3.0 * sigma_std;
  theta = exp(sigma/2) * theta_std;
}
model {
  sigma_raw ~ std_normal(); // implies sigma ~ normal(0, 3)
  theta_raw ~ std_normal(); // implies theta ~ normal(0, exp(sigma/2))
}
