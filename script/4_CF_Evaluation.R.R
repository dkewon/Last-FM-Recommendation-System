##Load packages
# install.packages("dplyr")
library(dplyr)
library(e1071)
# install.packages("tidyr")
library(tidyr)

# install.packages("recommenderlab")
library(recommenderlab)

###Set the working directory
## Set wd to location where you 
wd = "C:/Users/dkewon/Documents/RecommendationTool/Data Group Assignment"
setwd(wd)
getwd()
####Read in the data
artists <- read.csv("Artists.dat", sep="\t")
tags <- read.csv("tags.dat", sep="\t")
user_artists<- read.csv("user_artists.dat", sep="\t")
user_taggedartists <- read.csv("user_taggedartists.dat", sep="\t")

##Check uniaue users and artist IDs
#unique users
length(unique(user_artists$userID)) #1892
#unique artists/items
length(unique(user_artists$artistID)) #17632


summary(user_artists$weight)
hist(user_artists$weight) #right skewed
skewness(user_artists$weight)

#count <- ungroup(user_artists) %>% 
#group_by(userID) %>% 
#summarize(Count=n()) %>% 
#arrange(desc(Count))

#Mean_weight <- ungroup(user_artists) %>% 
#group_by(artistID) %>% 
#summarize(Mean_weight = mean(weight)) %>% 
#arrange(desc(Mean_weight))

#New_user_artists <- merge(user_artists, count, by.x='userID', by.y='userID', all.x=T)
#New_user_artists <- merge(user_artists, Mean_weight, by.x='userID', by.y='userID', all.x=T)
#New_user_artists$Mean_weight <- as.numeric(New_user_artists$Mean_weight)
#hist(New_user_artists$Mean_weight)
#str(New_user_artists)
#summary(New_user_artists$Mean_weight)

####Transform data to fix the skewness using log transformation
New_user_artists <- user_artists
New_user_artists$weight <- as.numeric(New_user_artists$weight)
New_user_artists$trans_weight<-log10(10*New_user_artists$weight) 
hist(New_user_artists$trans_weight)

str(New_user_artists)



summary(New_user_artists$trans_weight)

###Convert the dataframe into a wide matrix
names(New_user_artists)
New_user_artists <- New_user_artists[,c(1,2,4)]
New_user_artists$userID<- sprintf("%04d",New_user_artists$userID) 
New_user_artists$userID <-paste0('u',New_user_artists$userID)
New_user_artists$artistID<- sprintf("%05d",New_user_artists$artistID)
New_user_artists$artistID <-paste0('a',New_user_artists$artistID)

New_user_artists_wide <- spread(New_user_artists, key = artistID, value = trans_weight )
New_user_artists_wide[1:10,1:10]

New_user_artists_matrix <- data.matrix(New_user_artists_wide)
row.names(New_user_artists_matrix) <- New_user_artists_matrix[,1]
#drop first column
New_user_artists_matrix<- New_user_artists_matrix[,-1]
#add row names
row.names(New_user_artists_matrix) <- New_user_artists_wide[,1]
New_user_artists_matrix[1:10,1:10]

####Computing pearson correlation function
##split the data into train and test
num_rows <- nrow(New_user_artists_matrix) 
New_user_artists_matrix[is.na(New_user_artists_matrix)] <- 0

# split into 70/30, takes about 40 mins to run
set.seed(123) # Set a seed to have the same subsets every time 
# Define proportion to be in training set 
p <- 0.7
# Define observations to be in training set
training_locations <- sort(sample(num_rows,floor(p*num_rows)))
train_data <- New_user_artists_matrix[training_locations,]
test_data <- New_user_artists_matrix[-training_locations,]

# ### test function with small chunk of dataset
# ### this small chunk takes about 4 mins to run
p <- 0.99
# Define observations to be in training set
training_locations <- sort(sample(num_rows,floor(p*num_rows)))
train_data <- New_user_artists_matrix[training_locations,]
test_data <- New_user_artists_matrix[-training_locations,]

rownames(train_data)
rownames(test_data)
NN = 3
N = 10
onlyNew=TRUE

##1.Using a function

UserBasedCF <- function(train_data, test_data, N, NN, onlyNew=TRUE){
  
  ### similarity ###
  #Initialize an empty matrix
  
  # row.names(test_data) <- paste0('u',test_data[,1])
  # row.names(train_data) <- paste0('u',train_data[,1])
  similarity_matrix <- matrix(, nrow = nrow(test_data), ncol = nrow(train_data), 
                              dimnames = list(rownames(test_data), rownames(train_data)))
  
  ptm <- proc.time()
  ### pearson correlation calculation matrix
  for (i in 1:nrow(test_data)){
    for (j in 1:nrow(train_data)){
      r_xi <- test_data[i,]
      r_yi <- train_data[j,]
      r_xbar <- mean(test_data[i, ], na.rm=TRUE)
      r_ybar <- mean(train_data[j, ], na.rm=TRUE)
      
      sim_xy <- sum((r_xi-r_xbar)*(r_yi-r_ybar), na.rm=TRUE)/(sqrt(sum((r_xi-r_xbar)^2)) * sum((r_yi-r_ybar)^2))
      similarity_matrix[i, j] <- sim_xy
    }
    Time <- (proc.time() - ptm)
    print(i)
    print(Time)  
  }
  print("similarity calculation done")
  
  
  ### Nearest Neighbors ###
  similarity_matrix_NN <- similarity_matrix
  
  for (k in 1:nrow(similarity_matrix_NN)){
    crit_val <- -sort(-similarity_matrix_NN[k,])[NN]
    similarity_matrix_NN[k,] <- ifelse(similarity_matrix_NN[k,] >= crit_val, similarity_matrix_NN[k,], NA)
  }
  
  print("Nearest Neighbor selection done")
  ### Prediction ###
  # Prepare (intialize empty matrix)
  prediction <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                       dimnames=list(rownames(test_data), colnames(test_data)))
  prediction2 <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                        dimnames=list(rownames(test_data), colnames(test_data)))
  
  TopN <- matrix(, nrow=nrow(test_data), ncol=N, dimnames=list(rownames(test_data)))
  ### Numerator ###
  
  u = rownames(test_data)[1]
  
  for (u in rownames(test_data)){
    similarity_vector <- na.omit(similarity_matrix_NN[u, ])
    
    NN_norm <- train_data[rownames(train_data) %in% names(similarity_vector),]
    
    CM <- colMeans(train_data, na.rm=TRUE)
    for (l in 1:ncol(NN_norm)){
      NN_norm[,l] <- NN_norm[,l] - CM[l]
    }
    NN_norm[is.na(NN_norm)] <- 0
    
    # Numerator
    Num = similarity_vector %*% NN_norm
    
    #Prediction
    prediction[u, ] =  mean(test_data[u, ], na.rm=TRUE)  + (Num/sum(similarity_vector, na.rm=TRUE))
    
    
    if (onlyNew == TRUE){
      unseen <- names(test_data[u, test_data[u,]==0])
      prediction2[u, ] <- ifelse(colnames(prediction) %in% unseen, prediction[u, ], NA)
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
    
    TopN[u, ] <- names(-sort(-prediction2[u, ])[1:N])
    
  }
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  return(res)
}

######Check for results using the  function

ResultsIBCF <- UserBasedCF(train_data, test_data, N = 3, NN= 10, onlyNew=TRUE) # onlyNew = TRUE

prediction <- as.data.frame(ResultsIBCF$prediction)

# prediction onlyNew=FALSE
(-sort(prediction[1,]))[1:10]

TopN <- as.data.frame(ResultsIBCF$topN)
write.csv(TopN,'TopN.csv')

###### use proxy package
# 8105.66 sec = 2.25 hrs
# install.packages('proxy')
# library('proxy')
# 
# ptm <- proc.time()
# 
# row.names(New_user_artists_wide) <- paste0('u',New_user_artists_wide[,1])
# similarity_matrix_proxy <- as.matrix(simil(New_user_artists_wide, method="pearson"))
# 
# Time <- (proc.time() - ptm)
# print(Time)
# print("similarity calculation done")


###### Use recommenderlab
# recom <- Recommender(train, method = "UBCF")
# pred <- predict(test, test, n = 10)
# 
# getList(pred)
# pred@ratings

########MAE#########
#use test_data and predictions
tosee <- New_user_artists2[c(1,2:10)]

MAE <- function(prediction, real){
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    MAE = (sum( abs(prediction - real), na.rm = TRUE ) / (nrow(prediction) * ncol(prediction)) )
    return(MAE)
  }else{
    return("MAE is done")
  }
}

#to test
st <- proc.time()
User10 = UserBasedCF(train_data, test_data, N=10, NN=10, onlyNew=TRUE)
(proc.time() - st)
print("ex")
MAE(User10$prediction, test_data)
# 0.02154942

User17 = UserBasedCF(train_data, test_data, N=10, NN=17, onlyNew=TRUE)
MAE(User17$prediction, test_data)
# 0.02135579


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


########## Classification ##########
x <- Classification(User10$prediction, test_data, threshold = 2)
recall <- x$Recall
precision <- x$Precision


########## F1 Score ##########
F1Score <- function (recall, precision)
{2*((precision*recall)/(precision+recall))
}

F1Score(recall, precision)


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/


# #Classification?
# ###  Recall/Precision
# 
# algorithms <- list(
#   POPULAR = list(name = "POPULAR", param = NULL),
#   IBCF = list(name = "IBCF", param = NULL),
#   UBCF = list(name = "UBCF", param = NULL),
#   SVD = list(name = "SVD", param = NULL)
# )
# 
# 
# Classification <- function(prediction, real, threshold=NA, TopN=NA){
#   if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
#     # Threshold #
#     if (!is.na(threshold)){
#       TP = sum(ifelse(prediction >= threshold & real >= threshold, 1, 0), na.rm=T)
#       FP = sum(ifelse(prediction >= threshold & real < threshold, 1, 0), na.rm=T)
#       FN = sum(ifelse(prediction < threshold & real >= threshold, 1, 0), na.rm=T)
#       Recall = TP/(TP+FN)
#       Precision = TP/(TP+FP)
#       Class_Thres = list(Recall, Precision)
#       names(Class_Thres) = c("Recall", "Precision")
#     }
#     if (!is.na(TopN)){
#       TP = vector(, length = nrow(prediction))
#       FP = vector(, length = nrow(prediction))
#       FN = vector(, length = nrow(prediction))
#       
#       for (i in nrow(prediction)){
#         threshold_pred = -sort(-prediction[i, ])[TopN]
#         threshold_real = -sort(-real[i, ])[TopN]
#         TP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
#         FP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] < threshold_real, 1, 0), na.rm=T)
#         FN[i] = sum(ifelse(prediction[i, ] < threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
#       }
#       TP = sum(TP[i])
#       FP = sum(FP[i])
#       FN = sum(FN[i])
#       Recall = TP/(TP+FN)
#       Precision = TP/(TP+FP)
#       Class_TopN = list(Recall, Precision)
#       names(Class_TopN) = c("Recall", "Precision")
#     }
#     
#     
#     if (!is.na(threshold) & !is.na(TopN)){
#       Class = list(Class_Thres, Class_TopN)
#       names(Class) = c("Threshold", "TopN")
#     }else if (!is.na(threshold) & is.na(TopN)) {
#       Class = Class_Thres
#     }else if (is.na(threshold) & !is.na(TopN)) {
#       Class = Class_TopN
#     }else{
#       Class = "You have to specify the 'Threshold' or 'TopN' parameter!"
#     }
#     return(Class)  
#   }else{
#     return("Dimension of prediction are not equal to dimension of real")
#   }
# }
# 
# # Classification Item 10
# Classification(ResultsIBCF$prediction, test_data, threshold=3)
# # Classification Item 15
# Classification(ResultsUBCF$prediction, test_data, threshold=3)