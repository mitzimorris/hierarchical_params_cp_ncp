parameters {
  real sigma;
  vector[9] theta;
}
model {
  sigma ~ normal(0, 3);
  theta ~ normal(0, exp(sigma/2));
}
