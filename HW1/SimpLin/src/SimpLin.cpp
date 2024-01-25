#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List SimpLinCpp(arma::vec x, arma::vec y){
  
  int n       = x.n_elem;
  arma::mat X = arma::join_horiz(arma::ones(n), x); // design matrix
  int k       = X.n_cols;
  
  // Coefficient
  arma::vec coef   = arma::inv(arma::trans(X) * X) * arma::trans(X) * y;
  
  // Standard Errors
  arma::vec resids = y - X * coef;
  double sigma2    = arma::accu(arma::square(resids))/(n-k);
  arma::vec StdErr = arma::sqrt(sigma2 * arma::diagvec(arma::inv(arma::trans(X) * X)));
  
  // 95% confidence interval
  double quantile  = R::qt(0.975, n-k, 1, 0);
  arma::vec LB     = coef - quantile * StdErr;
  arma::vec UB     = coef + quantile * StdErr;
  arma::mat CI     = arma::join_horiz(LB, UB);
  
  Rcpp::List results;
  results["Coefficients"]     = coef;
  results["Std. Errors"]      = StdErr;
  // results["95% Lower Bound"] = LB;
  // results["95% Upper Bound"] = UB;
  results["95% CI"]           = CI;
  results["Residuals"]        = resids;
  results["Predicted Values"] = X * coef;
  return results;
}