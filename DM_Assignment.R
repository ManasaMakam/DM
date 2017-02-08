library(RODBC)
cn <- odbcDriverConnect(connection="Driver={SQL Server Native Client 11.0};server=HARSHIT\\SQLEXPRESS;database=DM_Assignment;trusted_connection=yes;")
data <- sqlFetch(cn, 'clean_data', colnames=FALSE)

hist(data[,1],main="Frequency plot of A2", xlab = "A2 - Clump Thickness")
hist(data[,2],main="Frequency plot of A3", xlab = "A3 - Uniformity of Cell Size")
hist(data[,3],main="Frequency plot of A4", xlab = "A4 - Uniformity of Cell Shape")
hist(data[,4],main="Frequency plot of A5", xlab = "A5 - Marginal Adhesion")
hist(data[,5],main="Frequency plot of A6", xlab = "A6 - Single Epithelial Cell Size")
hist(data[,6],main="Frequency plot of A7", xlab = "A7 - Bare Nuclei")
hist(data[,7],main="Frequency plot of A8", xlab = "A8 - Bland Chromatin")
hist(data[,8],main="Frequency plot of A9", xlab = "A9 - Normal Nucleoli")
hist(data[,9],main="Frequency plot of A10", xlab = "A10 - Mitoses")
hist(data[,10],main="Frequency plot of C", xlab = "C - Class")

summary(data)

var(data)

for(col in data){ 
  temp <- table(as.vector(col))
  print(names(temp)[temp == max(temp)])
  }
cor(data)