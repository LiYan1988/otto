library(methods)
library(data.table)
library(Rtsne)
setwd("C:\\Users\\lyaa\\Documents\\otto")
train = fread("train.csv", header = T, data.table = F)[,-1]
test = fread("test.csv", header = T, data.table = F)[,-1]
y = train[,ncol(train)]
y = as.integer(gsub('Class_', '', y))-1
train = train[,-ncol(train)]

x = rbind(train, test)
x = as.matrix(x)
x = matrix(as.numeric(x), nrow(x), ncol(x))
x = log(x+1)

x_tsne = Rtsne(x, check_duplicates = T, pca = TRUE, 
               perplexity=30, theta=0.5, dims=3, verbose = TRUE)

write.csv(x_tsne$Y, file='../tsne3all.csv', quote=FALSE, row.names=FALSE)