##data cleaning####
data <- read.csv("D:\\train.csv") 
library(naniar)
any_na(data) #check na
summary(data)
#remove numeric outlier: 5 samples
par(mfrow=c(1,4))
boxplot(data$bone_length)$out
boxplot(data$rotting_flesh)$out
boxplot(data$hair_length)$out
boxplot(data$has_soul)$out
data <- subset(data,bone_length<0.81700143&bone_length>0.1)
data <- subset(data,rotting_flesh<0.9&rotting_flesh>0.1)
par(mfrow=c(1,2))
boxplot(data$bone_length)$out
boxplot(data$rotting_flesh)$out

#normal ck
shapiro.test(data$bone_length)
shapiro.test(data$rotting_flesh)
shapiro.test(data$hair_length)
shapiro.test(data$has_soul)

#one-hot
features <- setdiff(names(data), "type")
library(vtreat)
library(dplyr)
onehot <- vtreat::designTreatmentsZ(data, features, verbose = FALSE)
new <- onehot %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)     
trainx <- vtreat::prepare(onehot, data, varRestriction = new) 
trainy <- data$type
#to python clustering
write.table(trainx[,2:11],file = "D:\\train_new.csv",sep=",",row.names = FALSE)

#from python
trainx <- read.csv("D:\\train_final.csv")
traindata <- cbind(trainx,trainy)
chk <- traindata %>% 
       group_by(group,trainy) %>% 
       summarise(count=n()) 
traindata <- subset(traindata,group<4) 
##svm####
#feature extraction
#compare by AAD & variance 
library(rminer) 
model <- fit(trainy~.,data=traindata,model="svm",
             kpar=list(sigma=0.1),C=1)
svm <- Importance(model,traindata,measure="variance")#AAD
L <- list(runs=1,sen=t(svm$imp),sresponses=svm$sresponses)
mgraph(L,graph="IMP",leg=names(traindata),col="gray",Grid=10)

#train & tune svm model
library(e1071)
#bone_length+has_soul+hair_length+rotting_flesh+group,
tune.model <- tune.svm(trainy~bone_length+has_soul+hair_length+rotting_flesh+group,
                       data=traindata,type="C-classification",
                       kernel="radial",gamma = 2^c(-8,-4,0,4),
                       range=list(cost = 2^c(-8,-4,-2,0)))
tune.model$best.parameters
tune.model$best.model
model <- svm(trainy~bone_length+has_soul+hair_length+rotting_flesh+group,
             data=traindata,type="C-classification",kernel="radial",
             cost=1,gamma=0.0625)

#predict
target <- read.csv("D:\\test.csv") 

features <- setdiff(names(target), "type")
onehot <- vtreat::designTreatmentsZ(target, features, verbose = FALSE)
new <- onehot %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)     
testx <- vtreat::prepare(onehot, target, varRestriction = new) 
#to python clustering
write.table(testx[,2:11],file = "D:\\test_new.csv",sep=",",row.names = FALSE)

#from python
test <- read.csv("D:\\test_final.csv") 
testx <- cbind(target[,1],test)
future <- predict(model,testx)
future <- as.data.frame(future)
final <- cbind(future,testx)
write.table(final[,1:2],file = "D:\\5_feature.csv",sep=",",row.names = FALSE)


