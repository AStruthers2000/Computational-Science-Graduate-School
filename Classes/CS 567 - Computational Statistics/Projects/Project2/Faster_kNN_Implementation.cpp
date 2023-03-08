#include <Rcpp.h>
#include <unordered_set>

#define DEBUG false

#if DEBUG
int debug_count = 0;
#endif

/* Function signatures */
// Core functions
float IE_xy(Rcpp::NumericVector data_y, Rcpp::NumericVector data_x, int k);
Rcpp::NumericVector kNN(Rcpp::NumericVector data, int k);

// Helper functions
void print_vector(Rcpp::NumericVector x);


/* Function Definitions */
Rcpp::NumericVector kNN(Rcpp::NumericVector data, int k)
{
  int N = data.size();
  if(k > N - 1) {
    k = N - 1;
  }
  
  Rcpp::NumericVector result(N);
  for(int i = 0; i < N; i++)
  {
    Rcpp::NumericVector diff = Rcpp::abs(data[i] - data);
    std::sort(diff.begin(), diff.end());
    
    float Ri = diff[k];
    result[i] = k / (N * 2 * Ri);
  }
  
#if DEBUG
  Rprintf("kNN result:       ");
  print_vector(result);
#endif
  
  return result;
}


// [[Rcpp::export]]
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k)
{
  Rcpp::NumericVector yval = Rcpp::unique(data_y);
  float result = 0;
  for(int i = 0; i < yval.size(); i++)
  {
    Rcpp::NumericVector x = data_x[data_y == yval[i]];

    Rcpp::NumericVector kNN_result = kNN(x, k);
    
    float IE = Rcpp::mean(kNN_result);
    float weight = (float)x.size() / (float)data_x.size();
    result += IE * weight;

#if DEBUG
    Rprintf("Condition vector: ");
    print_vector(x);
    
    Rprintf("IE[%d]     = %2.6f\n", i, IE);
    Rprintf("weight[%d] = %2.6f\n", i, weight);
    Rprintf("result     = %f\n", result);
    
    debug_count++;
    if(debug_count > 10)
    {
      break;
    }
#endif
  }
  return result;
}

// helper functions
void print_vector(Rcpp::NumericVector x)
{
  Rprintf("{");
  for(int i = 0; i < x.size() - 1; i++)
  {
    Rprintf("%2.4f ", x[i]);
  }
  Rprintf("%2.4f}\n", x[x.size() - 1]);
}