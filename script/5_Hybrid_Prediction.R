#####################
### Hybrid RecSys ###
#####################
library("recommenderlab")
library("tm")
library("SnowballC")
library("dbscan")
library("proxy")
library(Matrix)
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
############################################### Get the test data
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
############################################################
prediction_UBCF<- readRDS("prediction_UBCF.rds")
row_names  <- prediction_UBCF[,1]
prediction_UBCF<-data.frame(prediction_UBCF)
prediction_UBCF<- prediction_UBCF[,-1]
prediction_UBCF[1:10,1:10]
# rownames(prediction_UBCF)<-row_names

Prediction_Cluster<- readRDS("Prediction_Cluster.rds")
Prediction_Cluster[1:10,1:10]

prediction_UBCF_mat <-data.matrix(prediction_UBCF)
Prediction_Cluster_mat <- data.matrix(Prediction_Cluster)



####################
### Compute Mean ###
####################

prediction_UBCF_mat[is.na(prediction_UBCF_mat)] <- 0
Prediction_Cluster_mat[is.na(Prediction_Cluster_mat)] <- 0
zero_pos <- Prediction_Cluster==0 | prediction_UBCF==0
zero_pos[is.na(zero_pos)] <- TRUE
zero_pos[1:10,1:10]

prediction_UBCF_mat[1:10,1:10]
Prediction_Cluster_mat[1:10,1:10]

hybrid_mean<- matrix(, nrow = nrow(Prediction_Cluster_mat), ncol = ncol(Prediction_Cluster_mat))
hybrid_mean[zero_pos] <- pmax(prediction_UBCF_mat,Prediction_Cluster_mat)[zero_pos]
hybrid_mean[!zero_pos] <- ((prediction_UBCF+Prediction_Cluster)/2)[!zero_pos]
hybrid_mean[1:10,1:10]


### Transform list back to matrix with correct number of dimensions ###
# Hybrid_prediction_mean <- matrix(hybrid_mean, nrow=nrow(test_data), ncol=ncol(test_data))
rownames(hybrid_mean) <- rownames(test_data)
colnames(hybrid_mean) <- colnames(test_data)

####################
### Compute Max ###
####################
hybrid_max <- pmax(prediction_UBCF_mat, Prediction_Cluster_mat)
hybrid_max[1:10,1:10]

### Transform list back to matrix with correct number of dimensions ###
rownames(hybrid_max) <- rownames(test_data)
colnames(hybrid_max) <- colnames(test_data)


### Evaluate ###
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

Hybrid_prediction_mean_MAE<- MAE(hybrid_mean, test_data)
print(Hybrid_prediction_mean_MAE)
# 0.2948115

Hybrid_prediction_max_MAE<- MAE(hybrid_max, test_data)
# 0.5693818

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

#########################
x <- Classification(hybrid_mean, test_data, threshold=2)
print(x)
recall <- x$Recall
precision <- x$Precision

# $Recall
# [1] 0.3452675
# 
# $Precision
# [1] 0.1212647

########## F1 Score ##########
hybrid_mean_F1Score <- 2*((precision*recall)/(precision+recall))
print(hybrid_mean_F1Score)
# hybrid_mean_F1Score =  0.1794893
#########################
#########################
x <- Classification(hybrid_max, test_data, threshold=2)
print(x)
recall <- x$Recall
precision <- x$Precision

# $Recall
# [1] 1
# 
# $Precision
# [1] 0.02155063

########## F1 Score ##########
hybrid_max_F1Score <- 2*((precision*recall)/(precision+recall))
print(hybrid_max_F1Score)
# hybrid_mean_F1Score =  0.04219199
#########################