install.packages("Rcpp")
library(Rcpp)

setwd("~/My Documents/School/Grad School/Classes/CS 567 - Computational Statistics/Projects/Project2/Testing")
sourceCpp("hello.cpp")
hello()
add(1, 4)

setwd("~/My Documents/School/Grad School/Classes/CS 567 - Computational Statistics/Projects/Project2/TestDLL2/TestDLL/x64/Debug")
getwd()
sessionInfo()
dyn.load("TestDLL.dll")
info <- getNativeSymbolInfo("add_numbers", "TestDLL")
func <- .C("add_numbers", as.integer(1), as.integer(2), result=as.integer(0))
func$result
dyn.unload("TestDLL.dll")
