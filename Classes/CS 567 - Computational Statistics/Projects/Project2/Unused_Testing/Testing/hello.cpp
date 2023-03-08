#include <Rcpp.h>

// [[Rcpp::export]]
void hello()
{
	Rprintf("Hello world!\n");
}

// [[Rcpp::export]]
int add(int x, int y)
{
  return x+y;
}