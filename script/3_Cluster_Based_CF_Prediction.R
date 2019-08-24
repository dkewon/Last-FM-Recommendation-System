##Load packages

# install.packages("dplyr")

library(dplyr)

library(e1071)

# install.packages("tidyr")

library(tidyr)

install.packages("recommenderlab")

library(recommenderlab)

#install.packages("Matrix")

library(Matrix)



##############################################Set the working directory
wd = "C:/Users/dkewon/Documents/RecommendationTool/Data Group Assignment"
# wd = "C:/Users/mmajid1/Desktop/Recommendation Tools/Data-20190222"

setwd(wd)
getwd()



##############################################Read in the data

artists <- read.csv("Artists.dat", sep="\t")

tags <- read.csv("tags.dat", sep="\t")

user_artists<- read.csv("user_artists.dat", sep="\t")

user_taggedartists <- read.csv("user_taggedartists.dat", sep="\t")

#unique user_taggedartists artists/items
length(unique(user_taggedartists$artistID)) #12523


##Check unique users and artist IDs

#unique users
length(unique(user_artists$userID)) #1892

#unique artists/items
length(unique(user_artists$artistID)) #17632


##Check the distribution of the weights

summary(user_artists$weight)

hist(user_artists$weight) # histogram is right skewed

skewness(user_artists$weight)


##################################################Transform data to fix the skewness using log transformation
# subset, select only artists those appear in user_taggedartists
New_user_artists <- user_artists[user_artists$artistID %in% unique(user_taggedartists$artistID),]

New_user_artists$weight <- as.numeric(New_user_artists$weight)
New_user_artists$trans_weight<-log10(10*New_user_artists$weight) 

##round of the weight values
New_user_artists <- New_user_artists %>% mutate_at(vars(trans_weight), funs(round(., 2)))
hist(New_user_artists$trans_weight)

#str(New_user_artists)
summary(New_user_artists$trans_weight)







#####################################################Convert the dataframe into a wide matrix

##Preprocess data before transforming it into a wide matrix
##Pick only userid,artistid and new transformed weights
New_user_artists <- New_user_artists[,c(1,2,4)]

#for the purpose of fast execution randomly split the dataframe before tranposing only 1000 users were picked
# New_user_artists <- New_user_artists[sample(nrow(New_user_artists), 1000), ]

## transform all user id into 4 integer length
New_user_artists$userID<- sprintf("%04d",New_user_artists$userID) 

##add 'u' before all userid numbers eg u0002
New_user_artists$userID <-paste0('u',New_user_artists$userID)

## transform all artist id into 5 integer length
New_user_artists$artistID<- sprintf("%05d",New_user_artists$artistID)

##add 'a' before all artistid numbers eg a00002
New_user_artists$artistID <-paste0('a',New_user_artists$artistID)

############## Use spread function to transpose the data
New_user_artists_wide <- spread(New_user_artists, key = artistID, value = trans_weight )

#Preview the data
New_user_artists_wide[1:10,1:10]

#convert into a matrix
New_user_artists_matrix <- data.matrix(New_user_artists_wide)
row.names(New_user_artists_matrix) <- New_user_artists_matrix[,1]

#drop first column
New_user_artists_matrix<- New_user_artists_matrix[,-1]

#add row names
row.names(New_user_artists_matrix) <- New_user_artists_wide[,1]
New_user_artists_matrix[1:10,1:10]

# split data to have same users with other algorithms
num_rows <- nrow(New_user_artists_matrix) 

set.seed(123) # Set a seed to have the same subsets every time 

# Define proportion to be in training set 
p <- 0.7

# Define observations to be in training set
training_locations <- sort(sample(num_rows,floor(p*num_rows)))
# train_data <- New_user_artists_matrix[training_locations,]
test_data <- New_user_artists_matrix[-training_locations,]
#*** only test_data is plugged in to cluster based, then save the prediction result and compare with others.

data2 <-test_data
centers<- 200
iter <-100
########################################################################################################################################
### Cluster based CF as a function ###
######################################
ClusterBasedCF <- function(data, N, centers, iter, onlyNew=TRUE){
  ptm <- proc.time()
  data2 <- data
  
  # fill with average product rating
  colmeans <- colMeans(data2, na.rm=TRUE)
  # if there are still any NAs, fill with 0
  colmeans[is.na(colmeans)] <- 0
  
  for (j in colnames(data2)){
    data2[, j] <- ifelse(is.na(data2[ ,j]), colmeans[j], data2[, j])
  }
  

  
  km <- kmeans(data2, centers=centers, iter.max=iter)
  
  head(km$cluster)
  head(km$centers)
  
  
  # Statistics of the groups
  tab <- table(km$cluster)
  
  # Assign users to groups
  RES <- cbind(data, as.data.frame(km$cluster))
  
  # Calculate average ratings for everi cluster
  aggregation <- aggregate(RES, list(RES$"km$cluster"), mean, na.rm=T)
  aggregation <- aggregation[,-1]
  
  # Make a prediction
  users <- as.data.frame(RES$"km$cluster")
  users <- cbind(users, rownames(RES))
  colnames(users) <- c("km$cluster", 'rn')
  
  
  prediction = merge(users, aggregation, by="km$cluster")
  rownames(prediction) <- prediction$rn
  
  prediction  <- prediction[order(rownames(prediction)), -1:-2]
  
  prediction2 <- matrix(, nrow=nrow(prediction), ncol(prediction), 
                        dimnames=list(rownames(prediction), colnames(prediction)))
  colnames(prediction2) <- colnames(prediction)
  rownames(prediction2) <- rownames(prediction)
  
  for (u in rownames(prediction)){
    if (onlyNew == TRUE){
      unseen <- names(data[u, is.na(data[u,])])
      
      prediction2[u, ] <- as.numeric(t(ifelse(colnames(prediction) %in% unseen, prediction[u, ], as.numeric(NA))))
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
  }
  
  # TopN
  TopN <- t(apply(prediction, 1, function(x) names(head(sort(x, decreasing=TRUE), N))))
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  Time <- (proc.time() - ptm)
  print(Time)
  
  return(res)

} 
# ClusterBasedCF <- function(data, N, centers, iter, onlyNew=TRUE)
ResultsCluster <- ClusterBasedCF(test_data, 5, 100, 100, onlyNew=TRUE)
Prediction_Cluster <- as.data.frame(ResultsCluster$prediction)
Prediction_Cluster[1:10,1:10]
# *** only users appear in test_data will be savevd, so we can compare with other methods.
write.csv(Prediction_Cluster,'Prediction_Cluster.csv')

library(data.table)
saveRDS(Prediction_Cluster, file = "Prediction_Cluster.rds")

# # reload the table
# Prediction_Cluster <-fread(Prediction_Cluster.csv', header = T, sep = ',')
# Prediction_Cluster_<- readRDS("Prediction_Cluster.rds")

# prediction onlyNew=FALSE
(-sort(Prediction_Cluster[1,]))[1:10]
TopNcluster <- as.data.frame(ResultsCluster$topN)
write.csv(TopNcluster,'Prediction_Cluster_TopN.csv')


####################################################################################################
########MAE#########
MAE <- function(prediction, real){
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    MAE = (sum(abs(prediction - real), na.rm = TRUE) / (nrow(prediction) * ncol(prediction)) )
    print(MAE)
    return(MAE)
  }else{
    return("MAE is done")
  }
}
# replace NA NaN with 0
Prediction_Cluster[is.na(Prediction_Cluster)] <- 0
test_data[is.na(test_data)] <- 0

# change to matrix type
Prediction_Cluster_matrix <- data.matrix(Prediction_Cluster)
Prediction_Cluster_matrix[1:10,1:10]
test_data[1:10,1:10]

Cluster_MAE <- MAE(Prediction_Cluster_matrix, test_data)
# Cluster_MAE = 0.5600676


########## Recall/Precision ##########
Classification <- function(prediction, real, threshold=NA, TopN=NA){
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    # Threshold #
    if (!is.na(threshold)){
      TP = sum(ifelse(prediction >= threshold & real >= threshold, 1, 0), na.rm=T)
      FP = sum(ifelse(prediction >= threshold & real < threshold, 1, 0), na.rm=T)
      FN = sum(ifelse(prediction < threshold & real >= threshold, 1, 0), na.rm=T)
      Recall = TP/(TP+FN)
      Precision = TP/(TP+FP)
      Class_Thres = list(Recall, Precision)
      names(Class_Thres) = c("Recall", "Precision")
    }
    if (!is.na(TopN)){
      TP = vector(, length = nrow(prediction))
      FP = vector(, length = nrow(prediction))
      FN = vector(, length = nrow(prediction))
      
      for (i in nrow(prediction)){
        threshold_pred = -sort(-prediction[i, ])[TopN]
        threshold_real = -sort(-real[i, ])[TopN]
        TP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
        FP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] < threshold_real, 1, 0), na.rm=T)
        FN[i] = sum(ifelse(prediction[i, ] < threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
      }
      TP = sum(TP[i])
      FP = sum(FP[i])
      FN = sum(FN[i])
      Recall = TP/(TP+FN)
      Precision = TP/(TP+FP)
      Class_TopN = list(Recall, Precision)
      names(Class_TopN) = c("Recall", "Precision")
    }
    
    
    if (!is.na(threshold) & !is.na(TopN)){
      Class = list(Class_Thres, Class_TopN)
      names(Class) = c("Threshold", "TopN")
    }else if (!is.na(threshold) & is.na(TopN)) {
      Class = Class_Thres
    }else if (is.na(threshold) & !is.na(TopN)) {
      Class = Class_TopN
    }else{
      Class = "You have to specify the 'Threshold' or 'TopN' parameter!"
    }
    return(Class)  
  }else{
    return("Dimension of prediction are not equal to dimension of real")
  }
}

##########

x <- Classification(Prediction_Cluster_matrix, test_data, threshold=2)
print(x)
recall <- x$Recall
precision <- x$Precision
# $Recall
# [1] 1
# 
# $Precision
# [1] 0.02155131

########## F1 Score ##########
Cluster_F1Score <- 2*((precision*recall)/(precision+recall))
print(Cluster_F1Score)
# Cluster_F1Score = 0.0421933













