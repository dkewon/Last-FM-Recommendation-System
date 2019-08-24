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


#######################################Computing pearson correlation function

##split the data into train and test
num_rows <- nrow(New_user_artists_matrix) 

# replace NA with 0
New_user_artists_matrix[is.na(New_user_artists_matrix)] <- 0



# split into 70/30, takes about 30 mins to run

set.seed(123) # Set a seed to have the same subsets every time 

# Define proportion to be in training set 

p <- 0.7

# Define observations to be in training set

training_locations <- sort(sample(num_rows,floor(p*num_rows)))

train_data <- New_user_artists_matrix[training_locations,]
train_data[1:10,1:10]

test_data <- New_user_artists_matrix[-training_locations,]
test_data[1:10,1:10]
dim(test_data)

#rownames(train_data)

#rownames(test_data)
##define your number of recommendations N,nearest neighbour NN and OnlyNew (recommend only new stuff)
NN = 3
N = 10
onlyNew=TRUE



##1.Using a function

UserBasedCF <- function(train_data, test_data, N, NN, onlyNew=TRUE){
  
########## calculate the similarity
  
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
  hist(similarity_matrix)
  
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


ResultsUBCF <- UserBasedCF(train_data, test_data, N = 3, NN= 10, onlyNew=TRUE) # onlyNew = TRUE

prediction <- as.data.frame(ResultsUBCF$prediction)
prediction[1:10, 1:10]
write.csv(prediction,'prediction_UBCF.csv')

library(data.table)
prediction_UBCF <-fread('prediction_UBCF.csv', header = T, sep = ',')
saveRDS(prediction_UBCF, file = "prediction_UBCF.rds")

# # reload the table
prediction_UBCF<- readRDS("prediction_UBCF.rds")
prediction_UBCF <-prediction

# prediction onlyNew=FALSE

(-sort(prediction[1,]))[1:10]

TopN <- as.data.frame(ResultsUBCF$topN)
write.csv(TopN,'TopN.csv')

####################################################################################################
########MAE#########
MAE <- function(prediction, real){
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    MAE = (sum( abs(prediction - real), na.rm = TRUE ) / (nrow(prediction) * ncol(prediction)) )
    print(MAE)
    return(MAE)
  }else{
    return("MAE is done")
  }
}

# row_names  <- prediction_UBCF[,1]
# prediction_UBCF<-data.matrix(prediction_UBCF)
# prediction_UBCF<- prediction_UBCF[,-1]
# prediction_UBCF[1:10,1:10]
# row.names(prediction_UBCF)<-row_names

test_data[1:10,1:10]

UBCF_MAE <- MAE(prediction, test_data)
# UBCF_MAE = 0.02995557


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



x <- Classification(prediction, test_data, threshold=2)
print(x)
recall <- x$Recall
precision <- x$Precision

# $Recall
# [1] 0.02160912
# 
# $Precision
# [1] 0.8005908

########## F1 Score ##########
UBCF_F1Score <- 2*((precision*recall)/(precision+recall))
print(UBCF_F1Score)
# UBCF_F1Score = 0.04208238


