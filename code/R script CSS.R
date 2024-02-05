setwd("C:/Users/patri/Desktop/PATRICK/Università/Didattica/Corsi/Data Science/Computational Social Science/ESS9e03_1.stata")
library(tidyverse)
library(tidymodels)
library(ISLR2)
library(rsample)
library(MASS)
library(haven)
library(dplyr)
library(e1071)
library(class)
library(boot)
library(ggplot2)
library(reshape)
library(caret)
library(tree)
library(gbm)
library(modEvA)

dataCS <- read_dta('ESS9e03_1.dta')
dataCSS <- zap_labels(dataCS)                                         #Read dataset
                                                                      #Remove labels for ease of computations
#Data cleaning
vect <- dataCSS$iagmr                                                 
quant <- quantile(vect)
IQR <- IQR(vect)
dataCSS <- subset(dataCSS, (vect > quant[2]-(1.5*IQR)) & (vect < quant[4]+(1.5*IQR))) 
#Remove Outliers for iagmr

dataCSS <- subset(dataCSS, dataCSS$iagmr >= 18, select= c(rlgblg,iagmr,agea,gndr,bthcld,           
                                                          evmar,blgetmg,eduyrs,cntry, imptrad))      
#Remove those who plans to marry before 18 as it is illegal; filter keeping only relevant variables

boxplot(dataCSS)                                                      #Ignore outliers for independent variables 
dataCSS <- na.omit(dataCSS)                                           #Remove na's
dataCSS$cntry <- as.factor(dataCSS$cntry)
tr <- seq(1,length(dataCSS$iagmr),by=2)
ts <- seq(2,length(dataCSS$iagmr),by=2)

x_tr <- dataCSS[tr,]                                                   #Classic train-test 50/50 split
x_ts <- dataCSS[ts,]

cor.test(dataCS$eduyrs, dataCS$edulvlb, method = 'pearson')            #Used to decide between educational level and years spent in education.

tester <- x_ts$iagmr                                                   #Used for predict and classification
reg1 <- lm(iagmr~rlgblg +imptrad +blgetmg+agea+gndr+bthcld+evmar+eduyrs+cntry, data = dataCSS)
summary(reg1)                                                          #R2 equal to 0.138
plot(reg1)               
#Residuals vs Fitted are a straight line;Normal is bisector; scale-location is very near 1.0 residuals, 
#Residuals vs Leverage shows very low leverage, index of model stability.
lm.p <- predict(reg1, x_ts)
meanlm=mean(lm.p,na.rm=T)
lm.p <- replace(lm.p, is.na(lm.p),meanlm)                              #Replace NA in the predict with mean value.
table(round(lm.p),tester)                                              #Confusion Matrix
mean((round(lm.p)==tester))                                            #Accuracy. Everything works as intended!!! Accuracy = 0.137 Now, use 10 different splits.
BIC(reg1)                                                              #103'431
anova(reg1)    
x_ts['predict'] <- lm.p
mean((round(lm.p)-tester)**2)

reg2 <- lm(iagmr~rlgblg+blgetmg+imptrad+bthcld+evmar, data = dataCSS)
summary(reg2)                                                          #Very low R2
plot(reg2)               
lm.p <- predict(reg2, x_ts)
meanlm=mean(lm.p,na.rm=T)
mean((round(lm.p)==tester))                                            #The other variables have a higher impact.
BIC(reg1)                                                              #much higher AIC, it can be discarded.
anova(reg1)    

greg1 <- glm(iagmr~., data = dataCSS)
loocvglm <- cv.glm(x_tr, greg1, K = 100)                  
loocvglm$delta                                                         #10.9 as both raw and adjusted cross-validation estimator.

error <- NULL                                                          #Where we store our errors.
train.p <- 0.5                                                         #Training percentage
splits <- 10

set.seed(30)
for (i in 1:10){                                                       #Try 10 splits and calculate mean error
  spl = initial_split(dataCSS, strata = iagmr, prop = train.p)
  trainerc <- training(spl)
  testerc  <- testing(spl)
  test <- testerc$iagmr
  fit <- glm(iagmr~rlgblg+blgetmg+agea+gndr+bthcld+evmar+cntry+eduyrs, data = trainerc)
  prob <- predict(fit, testerc)
  table(round(prob), test)
  error[i] <- mean(round(prob)!=test)
} 
mean(error)                                                           #Error = 0.862 / Accuracy = 0.136; Consistent with the single lm.

iagmr_mean <- x_ts %>%  group_by(cntry) %>%  summarise_at(vars(iagmr), list(mean = mean))
p_mean <- x_ts %>%  group_by(cntry) %>%  summarise_at(vars(predict), list(meanp = mean))

iagmr_mean['p_mean'] <- p_mean$meanp
ggplot(iagmr_mean, aes(x=cntry, y=mean, fill=cntry)) + geom_col() + labs(x='Country', y='Average Ideal Age') + scale_y_continuous(breaks=seq(20,30,2)) + theme(text = element_text(size = 20)) + theme(legend.position="none")

summary(dataCSS$iagmr)
glm.fit2 <- glm(iagmr~rlgblg+blgetmg+bthcld+evmar, data = x_tr)        #This time, with only the 4 non-demographic variables.
plot(glm.fit2)
BIC(glm.fit2)
loocvglm2 <- cv.glm(x_tr, glm.fit2, K = 100)    
loocvglm2$delta                                                        #12.344 as both raw and ajudsted cross-validation estimator, less precise
 

regi <- lm(iagmr~rlgblg*blgetmg+imptrad+agea+gndr+bthcld+evmar+eduyrs+cntry, data = dataCSS)   #See whether rlgblg and blgetmg can have a relevant interaction.
anova(regi)                #Interaction isn't significant, should not be included.
summary(regi)
pred_int <- predict(regi, x_ts)

#LDA shouldn't be used, not a qualitative variable. Similarly, we cannot use knn as it is used for classification
#KNN classifiers tested but not implemented on final project. The result was 9 as the optimal k

#correlation matrix heatmap.
dataCSSm <- subset(dataCSS, select=c(rlgblg,iagmr,agea,gndr,bthcld,evmar,blgetmg,eduyrs,imptrad))  

cormat <- round(cor(dataCSSm),2)
melt_cormat <- melt(cormat)
ggplot(data = melt_cormat, aes(x=X1, y=X2,fill=value), font_size=2) + geom_tile() + scale_fill_gradient('value', low = "red", high = "green") + theme(text = element_text(size = 20))   

nB_model <- naiveBayes(iagmr~., data = dataCSS)
summary(nB_model)
nB_pred <- predict(nB_model, newdata=x_ts)
summary(nB_pred)
table(nB_pred, tester)   
mean(nB_pred == tester)                                                #Much more Accurate: 0.285. Can't compute AIC for it.

set.seed(30)
model <- train(iagmr ~ ., data = x_tr,method = 'knn')
plot(model)
model4 <- train(iagmr ~ .,data = x_tr,method = 'knn',preProcess = c("center", "scale"), trControl = ctrl)
plot(model4)

#Model Comparison:
glance(model) %>%   dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)

deg <- 4 # max polynomial degree
cv.error <- rep(0, deg) # initialise error vector
for (i in 1:deg) {
  glm.fit <- glm(iagmr ~ poly(eduyrs, i), data=dataCSS,)
  cv.error[i] <- cv.glm(dataCSS, glm.fit, K =100)$delta[1]         #Use grade 1
  }
cv.error
plot(cv.error, type='b', xlab='eduyrs index', ylab ='CV', ylim= c(12.2,12.4))

cv.error2 <- rep(0, deg) # initialise error vector
for (i in 1:deg) {
  glm.fit <- glm(iagmr ~ poly(agea, i), data=dataCSS,)
  cv.error[i] <- cv.glm(dataCSS, glm.fit, K =100)$delta[1]         #Use grade 1
}
cv.error2
plot(cv.error, type='b', xlab='agea index', ylab ='CV', ylim= c(12.2,12.4))

#Exclude country:
x <- subset(dataCSS, select=c(rlgblg,agea,gndr,bthcld,
                              evmar,blgetmg,eduyrs,imptrad))
x_ts <- dataCSS[ts,]
x_tr <- dataCSS[tr,]
tree <- tree(iagmr~., data=dataCSS)                                   #Regression Tree
summary(tree)
cv_tree <- cv.tree(tree)
pr_tree <- prune.tree(tree, best=cv_tree$size[which.min(cv_tree$dev)])
summary(pr_tree)                                                   #Pruned form
plot(pr_tree)                                        
text(pr_tree)
pr_pr_tree <- predict(pr_tree, newdata= x_ts)
mean((tester-round(pr_pr_tree))**2)                                #Pruned MSE:12.060
length(tester)


#You could add a random forest.
table(dataCSS$cntry)
y_tr <- x_tr$iagmr
y_ts <- x_ts$iagmr

iter <- seq(50,1000,50)
set.seed(12)
err_boost <- c()                                      
for (t in iter){                                                   #Boosting method --> exclude country!!! See which  number of trees is better.
  pr_boost <- gbm(iagmr~.,data=x_tr, distribution='gaussian',n.trees=t, interaction.depth=4)
  pred_boost <- predict(pr_boost, newdata=x_ts, n.trees=t)
  mse <- mean((y_ts-round(pred_boost))**2)
  err_boost <- append(err_boost, mse)
}
err_boost
plot(iter,err_boost, xlab='Boosting Iterations', ylab='MSE', type='b', col='blue')
pr_boost <- gbm(iagmr~.,data=dataCSS, distribution='gaussian',n.trees=300)    #300 trees as ideal; interaction depth gives no significant change.
summary(pr_boost)
pred_boost <- predict(pr_boost, newdata=x_ts, n.trees=100)
pred_boost
mse <- mean((y_ts-round(pred_boost))**2)                                   #Boosting mse: 11.81

#ModEvA model comparison
#Add all predicted model values to a new dataset with our y: lm.p; nB_pred; pr_pr_tree; pred_boost.     
#Add all models: reg1, nB_model, pr_tree, pr_boost
dataM <- data.frame(y = x_ts$iagmr, lm = lm.p, NB=nB_pred, Tree= pr_pr_tree, Boost = pred_boost)
models <- list(greg1, nB_model, pr_tree, pr_boost)
models[1]

plotGLM(model = models[1], xlab = "Logit (Y)", ylab = "Predicted probability", main = "Model plot")
#Can't be implemented as it only works on GLM functions, but I tried writing the script anyway.
