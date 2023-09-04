data {
  int<lower=1> J; // number of MCs
  int<lower=1> K; // number of items
  int<lower=1> N; // number of observations
  int<lower=1,upper=J> jj[N]; // user for observation n
  int<lower=1,upper=K> kk[N]; // item for observation n
  int X[N]; // Total attempts for user J
  int<lower=0> y[N]; // count of successful attempts for observation n
  
}

parameters {
  vector[K] delta;
  vector[K] alpha; // discrimination parameter
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_delta;
  vector[J] theta;
}

model {
  delta ~ normal(0, 1); // item intercept
  alpha ~ normal(0, 2); // discrimination/slope parameter
  sigma_alpha ~ normal(0,1);
  sigma_delta ~ normal(0,1);
  theta ~ normal(0, 1); // ideology/ability
  
  y ~ binomial_logit(X, delta[kk] + alpha[kk] .* theta[jj]);
}
