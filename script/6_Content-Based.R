#######################
### Content - Based ###
#######################  
library(tidyverse)
library(tidyr)
library(reshape2)
library(emulator)
library(apcluster)
library(dplyr)
library(proxy)
library(recommenderlab)

setwd('C:/Users/yliu10/Desktop/RMT/')
source("functions.R")
user_taggedartists <- read.csv("user_taggedartists.dat", sep="\t")
artists <- read.csv("Artists.dat", sep="\t")
tags <- read.csv("tags.dat", sep="\t")
user_artists <- read.csv("user_artists.dat", sep="\t")

#TOP Weight 500 ArtistID by mean
W <- aggregate(user_artists$weight, by=list(artistID = user_artists$artistID), FUN=mean)
WSort <- W[order(-W$x),]
TOP500 <- WSort[1:500,1]
TOP500 <- data.frame(TOP500)
colnames(TOP500) <- 'artistID'
UniqueArtist <- merge(user_artists,user_taggedartists, by='artistID')
UT <- merge(UniqueArtist,TOP500, by='artistID')
UT <- data.frame(unique(UT$artistID))
colnames(UT) <- 'artistID'


#######
#Data1#
#######
#Combining the dataset
user_taggedartists0 <- user_taggedartists[user_taggedartists$artistID %in% UT$artistID,]
Data <- merge(user_taggedartists0,tags, by='tagID')
Data$tagID  <- NULL
Data$day <- NULL
Data$month <- NULL
Data$year <- NULL
Data$userID <- NULL

#Remove duplicate
Data <-unique(Data)
Data <- Data[Data$artistID %in% UT$artistID,]

#Dcast
Data1 <- dcast(Data, artistID ~ tagValue, length, fill=0)
rownames(Data1) <- Data1$artistID
Data1$artistID <- NULL
Data <- as.matrix(Data1)

#######
#Data2#
#######
user_artist0 <- user_artists[user_artists$artistID %in% UT$artistID,]
user_artists1 <- user_artist0
user_artists1 <- user_artists1[user_artists1$artistID %in% UT$artistID,]
user_artists1$weight <- ifelse(user_artists1$weight<=50,1,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=250)&(user_artists1$weight>50),2,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=500)&(user_artists1$weight>250),3,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=1000)&(user_artists1$weight>500),4,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=2000)&(user_artists1$weight>1000),5,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=3000)&(user_artists1$weight>2000),6,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=5000)&(user_artists1$weight>3000),7,user_artists1$weight)
user_artists1$weight <- ifelse((user_artists1$weight<=10000)&(user_artists1$weight>5000),8,user_artists1$weight)
user_artists1$weight <- ifelse(user_artists1$weight>10000,9,user_artists1$weight)

#Long 
Data2 <- spread(data=user_artists1, key='artistID', value='weight')
rownames(Data2) <- Data2$userID
Data2$userID <- NULL
Data2 <- as.matrix(Data2)

##################
#Run ContentBased#
##################
CB <- ContentBased(Data1,Data2,3,10)
write.csv(CB$prediction, "prediction.csv")

########
#Result#
########

#MAE Evaluation
MAE <- MAE(CB$prediction, Data2)
print(MAE)


#RSME Evaluation
RSME <- RSME(CB$prediction, Data2)
print(RSME)


#Classification
Classification <- Classification(CB$prediction, Data2, 3.5)
print(Classification)

recall <- Classification$Recall
precision <- Classification$Precision

#F1Score
F1Score(recall, precision)
