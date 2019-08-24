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
#New_user_artists <- New_user_artists[sample(nrow(New_user_artists), 100), ]



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

class(New_user_artists_wide)

#***subset the data
### This will takes about 83 mins to complete ###
#Preview the data and splitting to allow computation. Two runs were done one with 2000 artists and another with 5000 artists
New_user_artists_wide <-New_user_artists_wide[,1:2001]
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



New_user_artists_matrix[is.na(New_user_artists_matrix)] <- 0



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



library(proxy)



ItemBasedCF <- function(train_data, test_data, N, NN, onlyNew=TRUE){
  
  
  
  similarity_matrix = matrix(, ncol=ncol(train_data), nrow=ncol(train_data), dimnames = list(colnames(train_data), colnames(train_data)))
  
  rowmeans = rowMeans(train_data)
  
  ptm <- proc.time()
  
  
  for (i in colnames(train_data)){
    
    for (j in colnames(train_data)){
      
      r_ui <- train_data[,i]
      
      r_uj <- train_data[,j]
      
      sim <- sum((r_ui- rowmeans)*(r_uj - rowmeans), na.rm=TRUE)/(sqrt(sum((r_ui-rowmeans)^2)) * sum((r_uj -rowmeans)^2))
      
      similarity_matrix[i, j] <- sim
      
      
      
    }
    
    Time <- (proc.time() - ptm)
    
    
    
    print(i)
    
    
    
    print(Time)  
    
    
    
  }
  
  
  
  print("Similarity calculation done")
  
  # Nearest Neighbor
  
  similarity_matrix_NN <- similarity_matrix
  
  
  
  for (k in 1:ncol(similarity_matrix_NN)){
    
    crit_val <- -sort(-similarity_matrix_NN[,k])[NN]
    
    similarity_matrix_NN[,k] <- ifelse(similarity_matrix_NN[,k] >= crit_val, similarity_matrix_NN[,k], NA)
    
  }
  
  similarity_matrix_NN[is.na(similarity_matrix_NN)] <- 0
  
  
  
  train_data[is.na(train_data)] <- 0
  
  
  
  test_data2 <- test_data
  
  test_data2[is.na(test_data2)] <- 0
  
  
  
  print("Nearest neighbor selection done")
  
  
  
  ### Prediction ###
  
  prediction <- matrix(, nrow=nrow(test_data), ncol=ncol(test_data), 
                       
                       dimnames=list(rownames(test_data), colnames(test_data)))
  
  prediction2 <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                        
                        dimnames=list(rownames(test_data), colnames(test_data)))
  
  TopN <- matrix(, nrow=nrow(test_data), N, dimnames=list(rownames(test_data)))
  
  
  
  for (u in rownames(test_data)){
    
    # Numerator
    
    Num <-  test_data2[u, ] %*% similarity_matrix_NN
    
    
    
    # Denominator
    
    Denom <- colSums(similarity_matrix_NN, na.rm=TRUE)
    
    
    
    # Prediction
    
    prediction[u, ] <- Num/Denom
    
    
    
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

ResultsIBCF <- ItemBasedCF(train_data, test_data, N = 3, NN= 10, onlyNew=TRUE) # onlyNew = TRUE

prediction <- as.data.frame(ResultsIBCF$prediction)

# write.csv(prediction,'prediction_IBCF.csv')

prediction_IBCF <-prediction

library(data.table)
# prediction_IBCF <-fread('prediction_IBCF.csv', header = T, sep = ',')
saveRDS(prediction_IBCF, file = "prediction_IBCF.rds")



(-sort(prediction[1,]))[1:10]



TopNIBCF <- as.data.frame(ResultsIBCF$topN)

write.csv(TopNIBCF,'TopNIBCF2.csv')



####################################################################################################

########MAE#########

MAE <- function(prediction, real){
  
  
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    
    MAE = (sum( abs(prediction - real), na.rm = TRUE ) / (nrow(prediction) * ncol(prediction)) )
    
    return(MAE)
    
  }else{
    
    return("MAE is done")
    
  }
  
}



UBCF_MAE <- MAE(prediction, test_data)

# UBCF_MAE = 0.06446814 when 2000 items are picked

# UBCF_MAE = 0.03257037 when 5000 items are picked              



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

#Recall   0.3493662 when 2000 artists are picked
#Recall   0.2747396 when 5000 artists are picked


precision <- x$Precision

#precision 0.7350689  when 2000 artists are picked
#precision 0.7149849  when 5000 artists are picked



########## F1 Score ##########

UBCF_F1Score <- 2*((precision*recall)/(precision+recall))

print(UBCF_F1Score)

# UBCF_F1Score = 0.4736258 when 2000 artists are picked
# UBCF_F1Score = 0.3969482 when 5000 artists are picked