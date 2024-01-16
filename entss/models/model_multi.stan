functions {
  real partial_sum_lpmf(array[] int slice_y,
                        int start, int end,
                        array[] int X,
                        vector theta,
                        vector delta,
                        vector alpha,
                        array[] int kk,
                        array[] int jj) {
    return binomial_logit_lupmf(slice_y |
                               X[start:end],
                               delta[kk[start:end]] + alpha[kk[start:end]] .* theta[jj[start:end]]);
  }
}
data {
  int<lower=1> J; // number of MCs
  int<lower=1> K; // number of items
  int<lower=1> N; // number of observations
  array[N] int<lower=1,upper=J> jj; // user for observation n
  array[N] int<lower=1,upper=K> kk; // item for observation n
  array[N] int X; // Total attempts for user J
  array[N] int<lower=0> y; // count of successful attempts for observation n
  int<lower=1> grainsize;
}

parameters {
  vector[K] delta;
  vector[K] alpha; // discrimination parameter
  vector[J] theta;
}

model {
  delta ~ normal(0, 1); // item intercept
  alpha ~ normal(0, 2); // discrimination/slope parameter
  theta ~ normal(0, 1); // ideology/ability
  
  target += reduce_sum(partial_sum_lupmf, y, grainsize, X, theta, delta, alpha, kk, jj);
}
