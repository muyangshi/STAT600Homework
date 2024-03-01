#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello World!" << std::endl;  
}

// [[Rcpp::export]]
double qx(double x){
  return(std::exp(-std::pow(std::abs(x),3)/3));
}

// [[Rcpp::export]]
arma::vec qx_vec(arma::vec x){
  return(arma::exp(-arma::pow(arma::abs(x),3)/3));
}

// [[Rcpp::export]]
arma::vec Rcpp_dnorm(Rcpp::NumericVector x, double mean = 0.0, double sd = 1.0, bool log = false){
  Rcpp::NumericVector d_rcpp = Rcpp::dnorm(x, mean, sd, log);
  arma::vec           d_arma = Rcpp::as<arma::vec>(wrap(d_rcpp));
  return(d_arma);
}

// [[Rcpp::export]]
arma::vec Rcpp_rnorm(int n = 1, double mean = 0.0, double sd = 1.0){
  Rcpp::NumericVector rsample_rcpp = Rcpp::rnorm(n, mean, sd);
  arma::vec           rsample_arma = Rcpp::as<arma::vec>(wrap(rsample_rcpp));
  return(rsample_rcpp);
}

// [[Rcpp::export]]
arma::vec w_star(arma::vec numerator, arma::vec denominator){
  return(numerator/denominator);
}

// [[Rcpp::export]]
arma::vec rejection_sampling(int n){
  arma::vec sample = arma::vec(n);
  int counter = 0;
  // int total_counter = 0;
  while(counter < n){
    double u = R::runif(0,1);
    double y = R::rnorm(0,1);
    if(u <= qx(y)/(3*R::dnorm(y, 0, 1, false))){
      sample[counter] = y;
      counter += 1;
    }
    // total_counter += 1;
    // if(total_counter > 10000){break;}
  }
  return(sample);
}

// [[Rcpp::export]]
double PhilippeRobert(arma::vec x){
  int x_length = x.n_elem;
  arma::vec x_no_first = x.subvec(1, x_length-1); // both endpoints inclusive
  arma::vec x_no_last = x.subvec(0, x_length-2);
  double num = arma::accu((x_no_first-x_no_last) % arma::pow(x_no_last,2) % qx_vec(x_no_last));
  double denom = arma::accu((x_no_first-x_no_last) % qx_vec(x_no_last));
  return(num/denom);
}