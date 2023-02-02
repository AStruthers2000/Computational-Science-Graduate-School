#Install and load necessary libraries
install.packages("ggplot2")

library(ggplot2)


#Read the data from file
#setwd("C://Users//Strut//Documents//School//Grad School//Classes//CS 567 - Computational Statistics//R_Workbook//Seminar_Codes//Data//")
#getwd()

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
  se <- sqrt(c(var(x), var(y)) / (length(x) - 2))
  sig <- cor.test(x, y)$p.value
  
  return(c(r = r, r2 = r2, covariance = c, significance = sig, standard_error = se))
}

# Define a function to calculate Spearman's rho
calc_spearman <- function(g) {
  x <- g[, 1]
  y <- g[, 2]
  rho <- cor(x, y, method = "spearman")
  c <- cov(x, y)
  se <- sqrt((2 * (2 * length(x) + 5)) / (9 * (length(x) - 1)))
  sig <- cor.test(x, y)$p.value
  return(c(rho = rho, covariance = c, significance = sig, standard_error = se))
}


#Separate the columns into pairs to be analyzed separately
g1 <- data[, c(1, 2)]
g2 <- data[, c(3, 4)]
g3 <- data[, c(5, 6)]


#Analyze the first two columns first
ggplot(g1, aes(x = g1[, 1], y = g1[, 2])) + geom_point() + xlab("Group 1 X Values") + ylab("Group 1 Y values") + ggtitle("Scatter Plot of Group 1")
p_r1 <- calc_pearson(g1)
c_r1 <- calc_spearman(g1)
p_r1
c_r1


#Analyze the second two columns
ggplot(g2, aes(x = g2[, 1], y = g2[, 2])) + geom_point() + xlab("Group 2 X Values") + ylab("Group 2 Y values") + ggtitle("Scatter Plot of Group 2")
p_r2 <- calc_pearson(g2)
c_r2 <- calc_spearman(g2)
p_r2
c_r2


#Analyze the third two columns
ggplot(g3, aes(x = g3[, 1], y = g3[, 2])) + geom_point() + xlab("Group 3 X Values") + ylab("Group 3 Y values") + ggtitle("Scatter Plot of Group 3")
p_r3 <- calc_pearson(g3)
c_r3 <- calc_spearman(g3)
p_r3
c_r3

