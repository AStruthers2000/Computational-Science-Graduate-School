#Install and load necessary libraries
install.packages("Hmisc")
install.packages("polycor")
install.packages("ggm")
install.packages("Rcmdr")

library(Hmisc)
library(ggplot2)
library(boot)
library(polycor)
library(ggm)
# library(Rcmdr)


#Read the data from file
setwd("C://Users//Strut//Documents//School//Grad School//Classes//CS 567 - Computational Statistics//R_Workbook//Seminar_Codes//Data//")
getwd()

data <- read.csv("seminartwodata.csv", header = FALSE)


#Define function for Pearson's R and R^2

calc_pearson <- function(x, y)
{
  r <- cor(x, y, method = "pearson")
  r2 <- r^2
  c <- cov(x, y)
  se <- sqrt(c(var(x), var(y)) / length(x) - 2)
  sig <- cor.test(x, y)$p.value
  
  return(c(r = r, r2 = r2, covar = c, sig = sig, se = se))
}


#Separate the columns into pairs to be analyzed separately
g1 <- data[, c(1, 2)]
g2 <- data[, c(3, 4)]
g3 <- data[, c(5, 6)]



