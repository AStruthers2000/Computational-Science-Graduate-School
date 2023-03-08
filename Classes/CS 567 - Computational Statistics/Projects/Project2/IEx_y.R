# kNN-approximated Probability Densities for Information Energy
# Authors - Prof. Yvonne Chueh, CWU


#  Input multivariate dataset and choose bivariate data to calculate information energy
#  and conditional information energy for the purpose of unilateral dependence measure.
#  Definition of information energy: Expected value of probability density function.
#  Here we define kNN as the function of input data and the choice of k value.

#------ kNN function
kNN <- function(data, k) {
  N <- length(data)
  # Sets the value of k to N-1 in case the prespecified value of k exceeds N-1.
  if(k > N-1) {
    k = N-1
  }
  else {
    k = k
  }
  # Start with blank density vector
  kNN <- c()
  for (i in 1:N) {
    # Determine the radius, or distance from the ith point in the vector to
    # kth nearest point.
    Ri <- sort(abs(data[i] - data[-i]))[k]
    # Append the kNN-approximated density for ith point to density vector
    kNN <- append(kNN, k/(N*2*Ri))
  }
  # Return vector of kNN-approximated densities
  kNN
}

#------ IE(X|Y) function
IEx_y <- function(data_x, data_y, k) {
  # Get the vector of unique values (levels) for conditional variable data_y
  yval <- unique(data_y)
  # Start with empty vectors for information energy and weight
  IE <- c()
  weight <- c()
  for (i in 1:length(yval)) {
    # Get the subset of main variable data_x conditioned on one value of data_y
    x <- data_x[data_y == yval[i]]
    # Calculate the conditional information energy using kNN approximation
    # formula described earlier and append to information energy vector.
    IE <- append(IE, mean(kNN(x,k)))
    # Calculate weight of data_x conditioned on one value of data_y and
    # append to weight vector.
    weight <- append(weight, length(x) / length(data_x))
  }
  # Return IE(X|Y)
  sum(IE * weight)
}

#---------------------- Experiments --------------------#

# -------- Experiment 1
x <- c(1,3,3,5,7,18,13,20,17,7)
y <- c(1,2,2,1,1,1,1,2,2,2)
IEx_y(x,y,2)

# -------- Experiment 2 form data file
# Read in data file
getwd()
setwd("/Users/DonaldD/Desktop/Yvonne_Project_2023") # Change working directory if necessary 
Data <- read.csv("Data.csv", skip = 2) # remove the first two rows 

#compare two rows withe different varying vlaues of k
attach(Data)
system.time({
  a <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 5000)
  b <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 7500)
  c <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 10000)
  d <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 12500)
  e <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 15000)
  f <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 17500)
  g <- IEx_y(MATHEFF[1:50000], MATINTFC[1:50000], 20000)
})
plot(seq(5000,20000,2500),c(a,b,c,d,e,f,g), xlab = "k", ylab = "IE(MATHEFF|MATINTFC)")
detach(Data)

