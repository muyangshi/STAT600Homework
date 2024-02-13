#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello World!" << std::endl;  
}

// [[Rcpp::export]]
double loglik_cauchy_cpp(double theta, arma::vec x){
  double result = -arma::accu(log(1 + arma::pow(x-theta,2)));
  return result;
}

// [[Rcpp::export]]
double dloglik_cauchy_cpp(double theta, arma::vec x){
  double result = 2*arma::accu((x - theta)/(1 + arma::pow(x - theta, 2)));
  return result;
}

// [[Rcpp::export]]
double ddloglik_cauchy_cpp(double theta, arma::vec x){
  double result = arma::accu((-2+2*(arma::pow(x-theta,2)))/(arma::pow(1 + arma::pow(x-theta,2),2)));
  return result;
}

int Bisection_cauchy_counter = 0;
int Newton_cauchy_counter = 0;
int Fisher_cauchy_counter = 0;
int Secant_cauchy_counter = 0;

// [[Rcpp::export]]
void print_counter(std::string counter){
  if(counter == "Bisection"){
    std::cout << "Total iterations: " << Bisection_cauchy_counter << std::endl;
  } else if (counter == "Newton"){
    std::cout << "Total iterations: " << Newton_cauchy_counter << std::endl;
  } else if (counter == "Fisher"){
    std::cout << "Total iterations: " << Fisher_cauchy_counter << std::endl;
  } else if (counter == "Secant"){
    std::cout << "Total iterations: " << Secant_cauchy_counter << std::endl;
  }
}
// [[Rcpp::export]]
void reset_counter(std::string counter){
  if(counter == "Bisection"){
    Bisection_cauchy_counter = 0;
  } else if (counter == "Newton"){
    Newton_cauchy_counter = 0;
  } else if (counter == "Fisher"){
    Fisher_cauchy_counter = 0;
  } else if (counter == "Secant"){
    Secant_cauchy_counter = 0;
  } else {
    std::cout << "wrong counter name" << std::endl;
  }
}


// [[Rcpp::export]]
double Bisection_cauchy_cpp(double a, double b, arma::vec dat, double eps=1e-8){
  Bisection_cauchy_counter += 1;
  double x = (a + b)/2;
  double a_new;
  double b_new;
  if(dloglik_cauchy_cpp(a, dat) * dloglik_cauchy_cpp(x, dat) <= 0.0){
    a_new = a;
    b_new = x;
  } 
  else { // dloglik_cauchy_cpp(a, dat) * dloglik_cauchy_cpp(x, dat) > 0.0
    a_new = x;
    b_new = b;
  } 
  double x_new = (a_new + b_new)/2;
  
  if(std::abs(x - x_new) < eps){
    print_counter("Bisection");
    reset_counter("Bisection");
    return x_new;
  } else {
    return Bisection_cauchy_cpp(a_new, b_new, dat, eps);
  }
}


// [[Rcpp::export]]
double Newton_cauchy_cpp(double x, arma::vec dat, double eps=1e-8){
  Newton_cauchy_counter += 1;
  if(ddloglik_cauchy_cpp(x, dat) == 0){
    print_counter("Newton");
    reset_counter("Newton");
    Rcpp::stop("l''(theta_hat) equals 0!");
  }
  double h = dloglik_cauchy_cpp(x, dat)/ddloglik_cauchy_cpp(x, dat);
  double x_new = x - h;
  if(std::abs(x - x_new) < eps){
    print_counter("Newton");
    reset_counter("Newton");
    return x_new;
  } else {
    return Newton_cauchy_cpp(x_new, dat, eps);
  }
}

// [[Rcpp::export]]
double FisherScoring_cauchy_cpp(double theta, arma::vec dat, double eps=1e-8){
  Fisher_cauchy_counter += 1;
  if(ddloglik_cauchy_cpp(theta, dat) == 0){
    print_counter("Fisher");
    reset_counter("Fisher");
    Rcpp::stop("l''(theta_hat) equals 0!");
  }
  double h = dloglik_cauchy_cpp(theta, dat)/(-ddloglik_cauchy_cpp(theta, dat));
  double theta_new = theta + h;
  if(std::abs(theta - theta_new) < eps){
    print_counter("Fisher");
    reset_counter("Fisher");
    return theta_new;
  } else {
    return FisherScoring_cauchy_cpp(theta_new, dat, eps);
  }
}

// [[Rcpp::export]]
double Secant_cauchy_cpp(double x0, double x1, arma::vec dat, double eps=1e-8){
  Secant_cauchy_counter += 1;
  double approx = (dloglik_cauchy_cpp(x1, dat) - dloglik_cauchy_cpp(x0, dat))/(x1 - x0);
  double x_new = x1 - dloglik_cauchy_cpp(x1, dat) / approx;
  if(std::abs(x_new - x1) < eps){
    print_counter("Secant");
    reset_counter("Secant");
    return x_new;
  } else{
    return Secant_cauchy_cpp(x1, x_new, dat, eps);
  }
}

// [[Rcpp::export]]
arma::vec dloglik_logit(arma::vec beta, arma::vec Y, arma::mat X){
  arma::vec XBeta = X * beta;
  arma::vec Yhat  = arma::exp(XBeta) / (1 + arma::exp(XBeta));
  return X.t() * (Y - Yhat);
}

// [[Rcpp::export]]
arma::mat ddloglik_logit(arma::vec beta, arma::vec Y, arma::mat X){
  arma::vec XBeta = X * beta;
  arma::vec D_vec = arma::exp(XBeta) / arma::pow(1 + arma::exp(XBeta), 2);
  arma::mat D     = arma::diagmat(D_vec);
  arma::mat H     = - X.t() * D * X;
  return H;
}


// [[Rcpp::export]]
Rcpp::List Newton_logit(arma::vec b0, arma::vec Y, arma::mat X, double eps){
  int iter = 1;
  arma::vec bt = b0;
  
  arma::vec dg    = dloglik_logit(bt, Y, X);
  arma::mat ddg   = ddloglik_logit(bt, Y, X);
  arma::vec b_new = bt - arma::inv(ddg) * dg;
  
  while(arma::norm(b_new - bt, 2) >= eps){
    if(iter == 1000){
      std::cout << bt << std::endl;
      Rcpp::stop("Has not converge after 1000 iterations.");
    }
    
    bt    = b_new;
    dg    = dloglik_logit(bt, Y, X);
    ddg   = ddloglik_logit(bt, Y, X);
    b_new = bt - arma::inv(ddg) * dg;

    iter += 1;
  }
  
  Rcpp::List results;
  results["iter"] = iter;
  results["Beta"] = bt;
  results["Hessian"] = ddg;
  return results;
}

// int add_cpp(int x, int y){return x+y;}
// 
// int minus_cpp(int x, int y){return x-y;}
// 
// typedef int (*funcPtr)(int x, int y);
// 
// // [[Rcpp::export]]
// Rcpp::XPtr<funcPtr> putFunPtrInXPtr(std::string fstr){ // this function takes a string argument, picks a function and returns it wrapped as an external pointer SEXP
//   if (fstr == "add_cpp")
//     return(Rcpp::XPtr<funcPtr>(new funcPtr(&add_cpp)));
//   else if (fstr == "minus_cpp")
//     return(Rcpp::XPtr<funcPtr>(new funcPtr(&minus_cpp)));
//   else
//     return Rcpp::XPtr<funcPtr>(R_NilValue);
// }
// 
// // [[Rcpp::export]]
// int callViaString(int x, int y, std::string funname) {
//   Rcpp::XPtr<funcPtr> xpfun = putFunPtrInXPtr(funname);
//   funcPtr fun = *xpfun;
//   int z = fun(x, y);
//   return (z);
// }
