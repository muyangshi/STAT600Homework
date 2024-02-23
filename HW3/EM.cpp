#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello World!" << std::endl;  
}

// [[Rcpp::export]]
arma::vec p_delta(arma::vec Y, int delta, double p, double lambda, double mu){
  arma::vec delta_0 = (1-p)*mu*arma::exp(-mu*Y);
  arma::vec delta_1 = p*lambda*arma::exp(-lambda*Y);
  if(delta == 0){
    return delta_0/(delta_1 + delta_0);
  }
  else if(delta == 1){
    return delta_1/(delta_1 + delta_0);
  }
  else{
    std::cout << delta << std::endl;
    Rcpp::stop("delta must be either int 0 or 1");
  }
}

// [[Rcpp::export]]
double Estep(arma::vec Y, double p, double lambda, double mu){
  return arma::accu(arma::log((1-p)*mu*arma::exp(-mu*Y)) * p_delta(Y, 0, p, lambda, mu) +
                    arma::log(p*lambda*arma::exp(-lambda*Y) * p_delta(Y, 1, p, lambda, mu)));
}

// [[Rcpp::export]]
double p_hat(arma::vec Y, double p, double lambda, double mu){
  double n = Y.n_elem;
  return (1/n)*arma::accu(p_delta(Y, 1, p, lambda, mu));
}

// [[Rcpp::export]]
double lambda_hat(arma::vec Y, double p, double lambda, double mu){
  return arma::accu(p_delta(Y, 1, p, lambda, mu))/arma::accu(Y % p_delta(Y, 1, p, lambda, mu));
}

// [[Rcpp::export]]
double mu_hat(arma::vec Y, double p, double lambda, double mu){
  double n = Y.n_elem;
  return (n - arma::accu(p_delta(Y, 1, p, lambda, mu)))/(arma::accu(Y % p_delta(Y, 0, p, lambda, mu)));
}

// [[Rcpp::export]]
Rcpp::List EM(arma::vec Y, arma::vec theta, double eps=1e-4){
  int iter = 1;
  double p;
  double lambda;
  double mu;
  double p_new;
  double lambda_new;
  double mu_new;
  arma::vec theta_t = arma::vec(theta);
  arma::vec theta_new = arma::vec(3);
  
  while(iter < 200){
    p      = theta_t[0];
    lambda = theta_t[1];
    mu     = theta_t[2];
    
    p_new      = p_hat(Y, p, lambda, mu);
    lambda_new = lambda_hat(Y, p, lambda, mu);
    mu_new     = mu_hat(Y, p, lambda, mu);
    
    theta_new[0] = p_new;
    theta_new[1] = lambda_new;
    theta_new[2] = mu_new;
    
    if(arma::norm(theta_t - theta_new, 2) <= eps){break;}
    
    theta_t = theta_new;

    iter += 1;
  }
  
  Rcpp::List results;
  results["theta"] = theta_new;
  return results;
}