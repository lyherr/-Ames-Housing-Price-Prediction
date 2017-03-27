train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)

#combine train and test data for preparation
test$SalePrice <- NA
train$isTrain <- 1
test$isTrain <- 0
combine.data <- rbind(train,test)

for(i in 1:dim(combine.data)[2]) {
  if(length(unique(combine.data[, i])) <= 2){
  print(colnames(combine.data)[i])
  }
}
combine.data$Utilities <- NULL  #low variance, providing less information


#deal with the missing value NA
sum.na <- sort(sapply(combine.data, function(x){sum(is.na(x))}), decreasing = TRUE)
#find the features that have missing value
names(which(sum.na != 0))
#if there are greater than 5% missing value then we could drop that feature
keep.col <- names(which(sum.na < dim(combine.data)[1] * 0.05))
combine.data <- combine.data[keep.col]



#missing values are caused by that it doesn't exist 
colnames(combine.data)[which(grepl("Bsmt", colnames(combine.data)))]
with(subset(combine.data, is.na(BsmtExposure)), summary(TotalBsmtSF))
with(subset(combine.data, is.na(BsmtExposure)), summary(BsmtFinSF1))
with(subset(combine.data, is.na(BsmtExposure)), summary(BsmtFinSF2))
with(subset(combine.data, is.na(BsmtExposure)), summary(BsmtUnfSF))

combine.data$BsmtExposure[which(is.na(combine.data$BsmtExposure))] <- 'Unf'
combine.data$BsmtFinType1[which(is.na(combine.data$BsmtFinType1))] <- 'Unf'
combine.data$BsmtFinType2[which(is.na(combine.data$BsmtFinType2))] <- 'Unf'
combine.data$BsmtQual[which(is.na(combine.data$BsmtQual))] <- 'Unf'
combine.data$BsmtCond[which(is.na(combine.data$BsmtCond))] <- 'Unf'

#MCAR missing completely at random
library(mice)
imp.combine.data <- mice(combine.data, m=1, printFlag=FALSE)
sort(sapply(complete(imp.combine.data), function(x) { sum(is.na(x)) }), decreasing=TRUE)
combine.data$MasVnrType <- as.factor(combine.data$MasVnrType)
combine.data$Electrical <- as.factor(combine.data$Electrical)
combine.data$MSZoning <- as.factor(combine.data$MSZoning)
combine.data$Functional <- as.factor(combine.data$Functional)
combine.data$KitchenQual <- as.factor(combine.data$KitchenQual)
combine.data$SaleType <- as.factor(combine.data$SaleType)
combine.data$Exterior1st <- as.factor(combine.data$Exterior1st)
combine.data$Exterior2nd <- as.factor(combine.data$Exterior2nd)
imp.combine.data <- mice(combine.data, m=1, method='cart', printFlag=FALSE)
sort(sapply(complete(imp.combine.data), function(x) { sum(is.na(x)) }), decreasing=TRUE)
combine.data <- complete(imp.combine.data)

combine.data[sapply(combine.data, is.character)] <- lapply(combine.data[sapply(combine.data, is.character)], as.factor)
#convert all catogorical value into factor 
sapply(combine.data, is.character) #check all the charaters are converted into factors


#then split the train and test from combine.data
train.Nosale <- combine.data[combine.data$isTrain==1,]  #SalePrice is deleted
test <- combine.data[combine.data$isTrain==0,]

train.Nosale$SalePrice <- train$SalePrice   #add the SalePrice back 
train <- train.Nosale

train <- subset(train, select = -c(Id,isTrain))
test <- subset(test, select = -c(isTrain)) #delete the Id and IsTrain

set.seed(1)
train.ind <- sample(1:dim(train)[1], dim(train)[1] * 0.7)
train.data <- train[train.ind, ]
test.data <- train[-train.ind, ]


plot(density(train.data$SalePrice))  #Right skewed
plot(density(log(train.data$SalePrice)))  #much normal, so using log(SalePrice)
#simple linear regression
mod.all <- lm(log(SalePrice) ~., data = train.data)
summary(mod.all)

#choose significant feature
mod.sig <- lm(log(SalePrice) ~ BsmtCond+MSZoning+BsmtFinSF1+KitchenQual+GarageArea+SaleType+LotArea
              +LandSlope+Neighborhood+Condition1+OverallQual
              +OverallCond+YearBuilt+RoofStyle+CentralAir+X1stFlrSF+X2ndFlrSF
              +FullBath+HalfBath+Fireplaces, data = train.data)
summary(mod.sig)
plot(mod.sig$residuals)
plot(mod.sig$fitted.values, mod.sig$residuals)

Sig.Prediction <- predict(mod.sig,test.data, type= "response" )
print(Sig.Prediction)

qqnorm(mod.sig$res)
qqline(mod.sig$res)


RMSE1 <- sqrt(mean(Sig.Prediction-log(test.data$SalePrice))^2)
#RMSE=0.00927

mod.sig.actual <- lm(log(SalePrice) ~ BsmtCond+MSZoning+BsmtFinSF1+KitchenQual+GarageArea+SaleType+LotArea
              +LandSlope+Neighborhood+Condition1+OverallQual
              +OverallCond+YearBuilt+RoofStyle+CentralAir+X1stFlrSF+X2ndFlrSF
              +FullBath+HalfBath+Fireplaces, data = train)         #using all train data to build model
actual.Prediction <- exp(predict(mod.sig.actual,test, type= "response" ))
sort(sapply(actual.Prediction, function(x){sum(is.na(x))}), decreasing = TRUE)
length(actual.Prediction)
submit <- data.frame(Id=test$Id,SalePrice=actual.Prediction)
write.csv(submit,file="3-22-17.csv",row.names=F)

#RMSE = 0.14, large difference from 0.009. Multicollinearity might exist

library(corrplot)
all_predictors <- subset(train,select=-c(SalePrice))
Predictors_classes <- sapply(all_predictors,function(x)class(x))
num_classes <- Predictors_classes[Predictors_classes=="integer"]
num_predictors <- train[,c("BsmtFinSF1","GarageArea","LotArea","OverallQual","YearBuilt","X1stFlrSF",
                           "X2ndFlrSF","FullBath","HalfBath","Fireplaces")]   #interger predictor in the mod.sig.actual model
corrplot(num_predictors, method = "number")

library(glmnet)
set.seed(1)
ind <- model.matrix(~., train.data[,-c(68)]) #delete Saleprice
dep <- log(train.data$SalePrice)

test_glmnet <- model.matrix(~., test.data[,-c(68)], rownames.force = NA)  #also delete Saleprice

fit <- glmnet(ind, dep)
plot(fit) 
plot(fit, label = T)
plot(fit, xvar = "lambda", label = T)
print(fit)

#choose the value of lambda
cvfit <- cv.glmnet(ind, dep)
plot(cvfit)  #could see the most appropriate lambda
cvfit$lambda.min # value of lambda gives minimal mean cross validated error
cvfit$lambda.1se # most regularized model such that error is within one std err of the minimum
#will choose lambda.min as the lambda
s=cvfit$lambda.min

#choose the value of alpha
foldid = sample(1:10, size=length(dep), replace=TRUE)
cv1 = cv.glmnet(ind, dep, foldid = foldid, alpha=1)
cv.5 = cv.glmnet(ind, dep, foldid = foldid, alpha=.5)
cv0 = cv.glmnet(ind, dep, foldid = foldid, alpha=0)
par(mfrow = c(1,2))
plot(cv1);plot(cv.5);plot(cv0)
plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name)
points(log(cv.5$lambda),cv.5$cvm,pch=19,col="grey")
points(log(cv0$lambda),cv0$cvm,pch=19,col="blue")
legend("topleft",legend=c("alpha= 1","alpha= .5","alpha= 0"),pch=19,col=c("red","grey","blue"))

fit.glmnet <- glmnet(x=ind, y=dep, alpha = 1, lambda = s)
glmnet.prediction <- predict(fit.glmnet, newx = test_glmnet, type = "response")

RMSE2 <- sqrt(mean(glmnet.prediction-log(test.data$SalePrice))^2)
RMSE2 #0.008554896 a little better than the previous unregularized model

# for the actual prediction 
ind.actual <- model.matrix(~., train[,-c(68)]) #delete Saleprice
dep.actual <- log(train$SalePrice)

test.actual.glmnet <- model.matrix(~., subset(test,select=-c(Id)))  #also delete Saleprice

actual.glmnet <- glmnet(x=ind.actual, y=dep.actual, alpha = 1, lambda = s)

glmnet.actual.prediction <- predict(actual.glmnet, newx = test.actual.glmnet, type = "response")

submit2 <- data.frame(Id=test$Id,SalePrice=exp(glmnet.actual.prediction))
write.csv(submit2,file="regularized 3-22-17.csv",row.names=F)


#RMSE = 0.131. Regularized model's performance is a little bit higher than previous one

#decision tree
library(rpart)
set.seed(1)
tree1 <- rpart(log(SalePrice) ~.-SalePrice, method = 'anova', data = train.data, 
               control=rpart.control(cp=0.001))
plot(tree1, uniform = TRUE)
text(tree1, cex = 0.5, xpd = TRUE)
printcp(tree1)
plotcp(tree1)
#choose the best tree size that could minimize xerror
bestcp <- tree1$cptable[which.min(tree1$cptable[,"xerror"]), "CP"]
tree.pruned <- prune(tree1, cp = bestcp)
tree.pruned
test.pred <- predict(tree.pruned, test.data)

library(rpart.plot)
prp(tree.pruned, faclen = 0, cex = 0.5)

RMSE3 <- sqrt(mean(test.pred-log(test.data$SalePrice))^2)
RMSE3 #0.00718 better performance than the regularized regression model


tree2 <- rpart(log(SalePrice) ~.-SalePrice, method = 'anova', data = train, 
               control=rpart.control(cp=0.001))
plot(tree2, uniform = TRUE)
text(tree2, cex = 0.5, xpd = TRUE)
printcp(tree2)
plotcp(tree2)
bestcp <- tree1$cptable[which.min(tree1$cptable[,"xerror"]), "CP"]
tree.pruned.actual <- prune(tree2, cp = bestcp)
tree.pruned.actual

test.tree <- cbind(test, SalePrice = 0)
tree.actual.pred <- predict(tree.pruned.actual, test.tree)
prp(tree.pruned.actual, faclen = 0, cex = 0.5)

tree.actual.pred <- exp(predict(tree.pruned.actual, test.tree))
tree.submit <- data.frame(Id=test$Id,SalePrice=tree.actual.pred)
write.csv(tree.submit,file="tree.3-22-17.csv",row.names=F)
#RMSE = 0.197 seems like decision tree is not a good option in this case


library(randomForest)
rf.formula <- paste("log(SalePrice) ~ .-SalePrice ")
rf.formula
rf <- randomForest(as.formula(rf.formula), data = train.data, importance = TRUE, ntree = 500)
getTree(rf, k = 1, labelVar = TRUE)

varImpPlot(rf) 
importance(rf, type = 1)
importanceOrder= order(-rf$importance[, "%IncMSE"])
names=rownames(rf$importance)[importanceOrder]
for (name in names[1:2]) {
  partialPlot(rf, train.data, eval(name), main=name, xlab=name, ylab=log(SalePrice))
}          #the top two important factor

rf.Predict <- predict(rf, test.data)
RMSE4 <- sqrt(mean(rf.Predict-log(test.data$SalePrice))^2)
RMSE4 #0.00518 even better performance
print(rf)

#try to predict the real value, use the whole train data
rf.actual <- randomForest(as.formula(rf.formula), data = train, importance = TRUE, ntree = 500)
rf.test <- cbind(test, SalePrice = 0)
rf.Predict.actual <- predict(rf.actual, rf.test)

rf.Predict.actual <- exp(predict(rf.actual, rf.test))
tree.submit <- data.frame(Id=test$Id,SalePrice=rf.Predict.actual)
write.csv(tree.submit,file="rf.3-22-17.csv",row.names=F)
#RMSE = 0.142 better than the decision tree but not as good as the regularized linear regression model

library(xgboost)
train.label <- log(train.data$SalePrice)
test.label <- log(test.data$SalePrice)
feature.matrix=model.matrix(~., train.data[,-c(68)]) #delete Saleprice

gbt <- xgboost(data = feature.matrix, 
                    label = train.label,
                    nround = 100,
                    max_depth = 5,
                    eta = 0.15,         
                    gamma = 0,
                    subsample = 0.9,
                    colsample_bytree = 0.75,
                    min_child_weight = 10,
                    max_delta_step = 8)
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = gbt)
head(importance)

xgb.plot.importance(importance[1:8,])

xgb_predict <- predict(gbt, model.matrix( ~ ., data = test.data))
RMSE5 <- sqrt(mean(xgb_predict-log(test.data$SalePrice))^2)
RMSE5  #only 0.0013 the best option for predicting the housing price

train.label.actual <- log(train$SalePrice)
feature.matrix.actual = model.matrix(~., train[,-c(68)])
gbt <- xgboost(data = feature.matrix.actual, 
               label = train.label.actual,
               nround = 100,
               max_depth = 5,
               eta = 0.15,         
               gamma = 0,
               subsample = 0.9,
               colsample_bytree = 0.75,
               min_child_weight = 10,
               max_delta_step = 8)
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = gbt)
head(importance)
xgb_predict <- predict(gbt, model.matrix( ~ ., data = subset(test, select=-c(Id))))
xgb_predict = exp(xgb_predict)
xbg.submit <- data.frame(Id=test$Id,SalePrice=xgb_predict)
write.csv(xbg.submit,file="xbg6.3-22-17.csv",row.names=F)

#RMSE=0.128 the best option for predicting


