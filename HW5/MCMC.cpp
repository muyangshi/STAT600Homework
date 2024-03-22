#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello World!" << std::endl;  
}

int rDiscreteUnif(double a, double b){
  double r = R::runif(a,b+1.0);
  int rDiscrete = int(std::floor(r));
  return(rDiscrete);
}

double rhalfnorm(double mean, double sd){
  return(std::abs(R::rnorm(mean, sd)));
}

double dhalfnorm(double x, double mean, double sd){
  if(x < 0){
    return(-INFINITY);
  }
  return(2.0*R::dnorm(x, mean, sd, false));
}

// [[Rcpp::export]]
arma::mat MCMC(int n_iters, arma::vec x, double lambda1_0, double lambda2_0, double alpha_0, int theta_0){
  arma::mat traceplot(n_iters, 4); // place to store params
  traceplot.fill(arma::datum::nan); // initialize to NaN
  
  traceplot(0,0) = lambda1_0;
  traceplot(0,1) = lambda2_0;
  traceplot(0,2) = alpha_0;
  traceplot(0,3) = theta_0;
  
  for(int iter = 1; iter < n_iters; iter++){
    // propose lambda1
    traceplot(iter,0) = R::rgamma(3 + arma::accu(x.subvec(0,traceplot(iter-1,3)-1)), 
                                  1/(traceplot(iter-1, 3) + traceplot(iter-1, 2)));
    
    // propose lambda2
    traceplot(iter,1) = R::rgamma(3 + arma::accu(x.subvec(traceplot(iter-1,3),111)),
                                  1/(112 - traceplot(iter-1, 3) + traceplot(iter-1, 2)));
    
    // propose alpha
    traceplot(iter,2) = R::rgamma(16, 
                                  1/(10 + traceplot(iter, 0) + traceplot(iter, 1)));
    
    // propose theta
    int theta_t = traceplot(iter-1, 3);
    int theta_n = rDiscreteUnif(1, 111);
    
    double log_lik_t = arma::accu(x.subvec(0, theta_t-1)) * std::log(traceplot(iter, 0))
                        - theta_t * traceplot(iter, 0)
                        + arma::accu(x.subvec(theta_t, 111)) * std::log(traceplot(iter, 1))
                        - (112 - theta_t) * traceplot(iter,1);
    double log_lik_n = arma::accu(x.subvec(0, theta_n-1)) * std::log(traceplot(iter, 0))
                        - theta_n * traceplot(iter, 0)
                        + arma::accu(x.subvec(theta_n, 111)) * std::log(traceplot(iter, 1))
                        - (112 - theta_n) * traceplot(iter,1);
    double lik_ratio = std::exp(log_lik_n - log_lik_t);
    double u = R::runif(0.0,1.0);
    if(lik_ratio > u){
      traceplot(iter, 3) = theta_n;
    } else{
      traceplot(iter, 3) = theta_t;
    }
  }
  return(traceplot);
}


// [[Rcpp::export]]
arma::mat MCMC_halfnorm(int n_iters, arma::vec x, double lambda1_0, double lambda2_0, double sigma, int theta_0){
  arma::mat traceplot(n_iters, 3); // place to store params
  traceplot.fill(arma::datum::nan); // initialize to NaN
  
  traceplot(0,0) = lambda1_0;
  traceplot(0,1) = lambda2_0;
  traceplot(0,2) = theta_0;
  
  for(int iter = 1; iter < n_iters; iter++){
    double log_lik_t;
    double log_lik_n;
    double lik_ratio;
    double u;
    
    // propose lambda1
    double lambda1_t = traceplot(iter-1, 0);
    // double lambda1_n = rhalfnorm(0, sigma);
    double lambda1_n = R::rnorm(lambda1_t, 0.1);
    if(lambda1_n < 0){
      traceplot(iter, 0) = lambda1_t;
    } else{
      log_lik_t = arma::accu(x.subvec(0, traceplot(iter-1,2)-1)) * std::log(lambda1_t)
                  - traceplot(iter-1, 2) * lambda1_t 
                  - std::pow(lambda1_t,2)/(2.0*std::pow(sigma,2));
      log_lik_n = arma::accu(x.subvec(0, traceplot(iter-1,2)-1)) * std::log(lambda1_n)
                  - traceplot(iter-1, 2) * lambda1_n
                  - std::pow(lambda1_n,2)/(2.0*std::pow(sigma,2));
      lik_ratio = std::exp(log_lik_n - log_lik_t);
      u = R::runif(0.0, 1.0);
      if(lik_ratio > u){
        traceplot(iter, 0) = lambda1_n;
      } else{
        traceplot(iter, 0) = lambda1_t;
      }
    }

    
    // propose lambda2
    double lambda2_t = traceplot(iter-1, 1);
    // double lambda2_n = rhalfnorm(0, sigma);
    double lambda2_n = R::rnorm(lambda2_t, 0.1);
    if(lambda2_n < 0){
      traceplot(iter, 1) = lambda2_t;
    } else {
      log_lik_t = arma::accu(x.subvec(traceplot(iter-1,2), 111)) * std::log(lambda2_t)
                  - (112.0 - traceplot(iter-1,2)) * lambda2_t
                  - std::pow(lambda2_t,2)/(2.0*std::pow(sigma,2));
      log_lik_n = arma::accu(x.subvec(traceplot(iter-1,2), 111)) * std::log(lambda2_n)
                  - (112.0 - traceplot(iter-1,2)) * lambda2_n
                  - std::pow(lambda2_n,2)/(2.0*std::pow(sigma,2));
      lik_ratio = std::exp(log_lik_n - log_lik_t);
      u = R::runif(0.0, 1.0);
      if(lik_ratio > u){
        traceplot(iter, 1) = lambda2_n;
      } else{
        traceplot(iter, 1) = lambda2_t;
      }      
    }
    
    // propose theta
    int theta_t = traceplot(iter-1, 2);
    int theta_n = rDiscreteUnif(1, 111);
    
    log_lik_t = arma::accu(x.subvec(0, theta_t-1)) * std::log(traceplot(iter, 0))
                - theta_t * traceplot(iter, 0)
                + arma::accu(x.subvec(theta_t, 111)) * std::log(traceplot(iter, 1))
                - (112 - theta_t) * traceplot(iter,1);
    log_lik_n = arma::accu(x.subvec(0, theta_n-1)) * std::log(traceplot(iter, 0))
                - theta_n * traceplot(iter, 0)
                + arma::accu(x.subvec(theta_n, 111)) * std::log(traceplot(iter, 1))
                - (112 - theta_n) * traceplot(iter,1);
    lik_ratio = std::exp(log_lik_n - log_lik_t);
    u = R::runif(0.0,1.0);
    if(lik_ratio > u){
      traceplot(iter, 2) = theta_n;
    } else{
      traceplot(iter, 2) = theta_t;
    }
  }
  return(traceplot);
}