data {
  int<lower=1> J; // number of MCs
  int<lower=1> K; // number of items
  int<lower=1> N; // number of observations
  array[N] int<lower=1,upper=J> jj; // user for observation n
  array[N] int<lower=1,upper=K> kk; // item for observation n
  array[N] int X; // Total attempts for user J
  array[N] int<lower=0> y; // count of successful attempts for observation n
  int<lower=1> G; // number of groups or parties
  array[J] int<lower=1, upper=G> gg; // group for observation n
  
}

parameters {
  vector[K] delta;
  vector[K] alpha; // discrimination parameter
  vector[J] theta;
  vector[G] mu_theta;
  real<lower=0> sigma_theta;
}

model {
  // hyperpriors
  sigma_theta ~ normal(.3, .1);
  mu_theta ~ normal(0, 1);
  
  //priors
  delta ~ normal(0, 1); // item intercept
  alpha ~ normal(0, 1); // discrimination/slope parameter
  theta ~ normal(mu_theta[gg], sigma_theta); // ideology/ability
  
  y ~ binomial_logit(X, delta[kk] + alpha[kk] .* theta[jj]);
}
