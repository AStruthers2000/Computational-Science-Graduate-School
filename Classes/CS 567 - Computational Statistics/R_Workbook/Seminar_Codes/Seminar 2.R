#Author: Andrew Struthers (Student ID 41371870)
#Date: 1/31/2023
#Honor code: I pledge that I have neither given nor received help from anyone 
#            other than the instructor or the TAs for all work components included here. 
#            -- Andrew

#Install and load necessary libraries
install.packages("ggplot2")
install.packages("boot")

library(ggplot2)
library(boot)


#Read the data from file
#setwd("C://Users//Strut//Documents//School//Grad School//Classes//CS 567 - Computational Statistics//R_Workbook//Seminar_Codes//Data//")
#getwd()

setwd("~/My Documents/School/Grad School/Classes/CS 567 - Computational Statistics/R_Workbook/Seminar_Codes")
setwd(file.path(getwd(), "Data"))
getwd()
data <- read.csv("seminartwodata.csv", header = FALSE)


#Define function for Pearson's R and R^2
calc_pearson <- function(g)
{
  x <- g[, 1]
  y <- g[, 2]
  r <- cor(x, y, method = "pearson")
  r2 <- r^2
  c <- cov(x, y)
  se <- sqrt(1/(length(x)-2)) * sqrt((1-r^2)/r^2)
  sig <- cor.test(x, y)$p.value
  
  return(c(r = r, r2 = r2, covariance = c, significance = sig, standard_error = se))
}

#Define a function to calculate Spearman's rho
calc_spearman <- function(g) {
  x <- g[, 1]
  y <- g[, 2]
  rho <- cor(x, y, method = "spearman")
  c <- cov(x, y)
  se <- 1/sqrt(length(x)-3)
  sig <- cor.test(x, y, method = "spearman")$p.value
  return(c(rho = rho, covariance = c, significance = sig, standard_error = se))
}

# Define a function to calculate Pearson's R and R^2 using bootstrapping
calc_pearson_boot <- function(x, y, R) {
  boot_result <- boot(data = cbind(x, y), 
                      statistic = function(d, i) cor(d[i, 1], d[i, 2], method = "pearson"), 
                      R = R)
  r <- boot_result$t[1]
  sig <- boot.ci(boot_result, type = "bca")$bca[4]
  return(c(r = r, significance = sig))
}

# Define a function to calculate Spearman's rho using bootstrapping
calc_spearman_boot <- function(x, y, R) {
  boot_result <- boot(data = cbind(x, y), 
                      statistic = function(d, i) cor(d[i, 1], d[i, 2], method = "spearman"), 
                      R = R)
  rho <- boot_result$t[1]
  sig <- boot.ci(boot_result, type = "bca")$bca[4]
  return(c(rho = rho, significance = sig))
}


#Separate the columns into pairs to be analyzed separately
g1 <- data[, c(1, 2)]
g2 <- data[, c(3, 4)]
g3 <- data[, c(5, 6)]

bootstrap_resampling <- 1000

#Analyze the first two columns first
ggplot(g1, aes(x = g1[, 1], y = g1[, 2])) + geom_point() + xlab("Group 1 X Values") + ylab("Group 1 Y values") + ggtitle("Scatter Plot of Group 1")
p_r1 <- calc_pearson(g1)
pb_r1 <- calc_pearson_boot(g1[, 1], g1[, 2], bootstrap_resampling)
c_r1 <- calc_spearman(g1)
cb_r1 <- calc_spearman_boot(g1[, 1], g1[, 2], bootstrap_resampling)
p_r1
pb_r1
c_r1
cb_r1

# ===== Testing results of the functions by using the "pure" versions of the library calls =====
cor(g1, use = "complete.obs", method = 'pearson')
cor.test(g1[, 1], g1[, 2])
cor.test(g1[, 1], g1[, 2], method = "spearman")
b_result <- boot(g1, statistic = function(d, i) cor(d[i, 1], d[i, 2], method = "pearson"), R = bootstrap_resampling)
b_result

#Analyze the second two columns
ggplot(g2, aes(x = g2[, 1], y = g2[, 2])) + geom_point() + xlab("Group 2 X Values") + ylab("Group 2 Y values") + ggtitle("Scatter Plot of Group 2")
p_r2 <- calc_pearson(g2)
pb_r2 <- calc_pearson_boot(g2[, 1], g1[, 2], bootstrap_resampling)
c_r2 <- calc_spearman(g2)
cb_r2 <- calc_spearman_boot(g2[, 1], g1[, 2], bootstrap_resampling)
p_r2
pb_r2
c_r2
cb_r2

#Analyze the third two columns
ggplot(g3, aes(x = g3[, 1], y = g3[, 2])) + geom_point() + xlab("Group 3 X Values") + ylab("Group 3 Y values") + ggtitle("Scatter Plot of Group 3")
p_r3 <- calc_pearson(g3)
pb_r3 <- calc_pearson_boot(g3[, 1], g1[, 2], bootstrap_resampling)
c_r3 <- calc_spearman(g3)
cb_r3 <- calc_spearman_boot(g3[, 1], g3[, 2], bootstrap_resampling)
p_r3
pb_r3
c_r3
cb_r3


