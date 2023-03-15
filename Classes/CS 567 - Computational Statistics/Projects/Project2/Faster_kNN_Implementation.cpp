#include <Rcpp.h>

//debugging information, setting DEBUG to true enables some print statements during execution
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

/*
  kNN function, which takes a vector of floats and a k value
  returns a vector of the kNN weights for each element of data.
  uses the built in std::sort algorithm to sort the vector elements 
  by their absolute distance to the i'th element in data
*/
Rcpp::NumericVector kNN(Rcpp::NumericVector data, int k)
{
  int N = data.size();

  //set the value of k to N-1 in the case where k > N-1
  if(k > N - 1) {
    k = N - 1;
  }
  
  //blank density vector
  Rcpp::NumericVector result(N);


  for(int i = 0; i < N; i++)
  {
    //determines the distance from the ith point in the vector
    Rcpp::NumericVector dist = Rcpp::abs(data[i] - data);

    //sorts the distances in ascending order of closeness to the ith element
    std::sort(dist.begin(), dist.end());
    
    //grab the kth element of the distance results
    //notice, we are indexing the dist vector by k, instead of k-1
    //we do this because, when calculating the dist vector, we subtract each element 
    //of data by data[i]. this means that we will have one '0', as a result of data[i] - data[i]
    //we don't remove the ith element first, so to counteract this indexing, we would
    //expect to grab dist[k + 1], but because indexes in C++ start at 0, we want 
    //dist[(k-1) + 1] = dist[k]
    float Ri = dist[k];

    //append the kNN-approximated density for the ith point to the density vector
    result[i] = k / (N * 2 * Ri);
  }
  
#if DEBUG
  Rprintf("kNN result:       ");
  print_vector(result);
#endif
  //return our kNN-approximated densities vector  
  return result;
}


/*
  this is the only function that is exposed to R (see Rcpp::export)
  this function is optimized relative to the provided R code by 
  using vectorized unique() as well as storing the cumulative result
  each iteration instead of creating an array of IE values and an array
  of weight values. this saves lots of space in memory and speeds up calculations
*/
// [[Rcpp::export]]
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k)
{
  //use vectorization to find the unique subset of conditional y-values
  Rcpp::NumericVector yval = Rcpp::unique(data_y);

  //store the result in a float instead of creating two empty IE and weight vectors
  float result = 0;
  for(int i = 0; i < yval.size(); i++)
  {
    //use vector logic to find the subset of x data that corresponds to this unique y-value
    Rcpp::NumericVector x = data_x[data_y == yval[i]];

    //calculate the conditional information energy using the kNN approximation
    Rcpp::NumericVector kNN_result = kNN(x, k);
    float IE = Rcpp::mean(kNN_result);

    //calculate the weight of the conditioned x vector (conditioned by the unique y-value)
    //on the overall dataset
    float weight = (float)x.size() / (float)data_x.size();

    //multiply the information energy by the weight and increase the result
    //calculating the result incrementally instead of storing IE and weight in two 
    //different vectors decreases the memory usage and number of instructions required
    //to calculate the same result
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
/*
  standard way to print out an array, used in a lot of code debugging
*/
void print_vector(Rcpp::NumericVector x)
{
  Rprintf("{");
  for(int i = 0; i < x.size() - 1; i++)
  {
    Rprintf("%2.4f ", x[i]);
  }
  Rprintf("%2.4f}\n", x[x.size() - 1]);
}