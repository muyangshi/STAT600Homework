SimpLinR <- function(x,y) {
  if(length(x) != length(y)){
    stop('x and y are of different length!')
  }
  SimpLinCpp(x, y)
}