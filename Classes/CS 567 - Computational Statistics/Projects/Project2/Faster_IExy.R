install.packages("Rcpp")
library("Rcpp")

setwd("~/My Documents/School/Grad School/Classes/CS 567 - Computational Statistics/Projects/Project2")

sourceCpp("Faster_kNN_Implementation.cpp")

# testing values, output should be "0.03535664"
x <- c(1, 3, 3, 5, 7, 18, 13, 20, 17, 7)
y <- c(1, 2, 2, 1, 1, 1,  1,  2,  2,  2)
test_val <- IE_xy(x, y, 2)


Data <- read.csv("Data.csv", skip = 2)

attach(Data)
system.time({
  a <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 5000)
  b <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 7500)
  c <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 10000)
  d <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 12500)
  e <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 15000)
  f <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 17500)
  g <- IE_xy(MATHEFF[1:50000], MATINTFC[1:50000], 20000)
})
plot(seq(5000,20000,2500),c(a,b,c,d,e,f,g), xlab = "k", ylab = "IE(MATHEFF|MATINTFC)")
detach(Data)

