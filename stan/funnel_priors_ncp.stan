parameters {
  real y_std;
  vector[9] x_std;
}
transformed parameters {
  real y = 3.0 * y_std;
  vector[9] x = exp(y/2) * x_std;
}
model {
  y_std ~ std_normal(); // implies y ~ normal(0, 3)
  x_std ~ std_normal(); // implies x ~ normal(0, exp(y/2))
}
