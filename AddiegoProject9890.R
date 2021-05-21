library(glmnet)
library(randomForest)
library(titoc)
rm(list = ls())

soccer = read.csv("dataset_soccer_ng2.csv")


soccer[is.na(soccer)] = 0
soccer$Name = NULL
soccer$Jersey.Number = NULL
soccer$Club = NULL
soccer$Position = NULL
soccer$Nationality = NULL



#Passes.per.match

lam.las = c(seq(0.01,1,length=100)) 
lam.rid = lam.las*5

err.matrix = as.data.frame(matrix(rep(0),nrow = 100, ncol = 8))
colnames(err.matrix) = c("lasso.train", "lasso.test", "elnet.train", 
                    "elnet.test", "ridge.train", "ridge.test", "rf.train", "rf.test")

data = soccer

for (i in 1:100){
  # split dataset
  set.seed (as.character(sample(1:1000), 1))
  s=sample(1:nrow(data), (nrow(data)*0.8))
  train = data[s,]
  test = data[-s,]
  
  y.train = as.matrix(train$Passes.per.match)
  x.train = as.matrix(subset(train, select = -Passes.per.match))
  
  y.test = as.matrix(test$Passes.per.match)
  x.test = as.matrix(subset(test, select = -Passes.per.match))
  
  
  # RIDGE
  start = Sys.time()
  
  cv.ridge = cv.glmnet(x.train, y.train, alpha = 0, lambda = lam.rid, nfolds = 10) 
  
  endcv = Sys.time()
  
  fit.ridge = glmnet(x.train, y.train, alpha = 0, lambda = cv.ridge$lambda.min)
  yhat.train.ridge = predict(fit.ridge, newx = x.train, type = "response") 
  yhat.test.ridge = predict (fit.ridge, newx = x.test, type = "response")
  
  rsqu.train.ridge = 1 - (mean((y.train - yhat.train.ridge)^2) / mean((y.train - mean(y.train))^2))
  rsqu.test.ridge = 1 - (mean((y.test - yhat.test.ridge)^2) / mean((y.test - mean(y.test))^2))
  
  err.matrix$ridge.train[i] = rsqu.train.ridge
  err.matrix$ridge.test[i] = rsqu.test.ridge
  print("Ridge done ")
  
  end = Sys.time()
  tcv.ridge = endcv - start
  t.ridge = end - start
  
  # ELNET
  start = Sys.time()
  
  cv.elnet = cv.glmnet(x.train, y.train, alpha = 0.5, lambda = lam.las, nfolds = 10)
  endcv = Sys.time()
  
  fit.elnet = glmnet(x.train, y.train, alpha = 0.5, lambda = cv.elnet$lambda.min)
  yhat.train.elnet = predict(fit.elnet, newx = x.train, type = "response") 
  yhat.test.elnet = predict (fit.elnet, newx = x.test, type = "response")
  
  rsqu.train.elnet = 1 - (mean((y.train - yhat.train.elnet)^2) / mean((y.train - mean(y.train))^2))
  rsqu.test.elnet = 1 - (mean((y.test - yhat.test.elnet)^2) / mean((y.test - mean(y.test))^2))
  
  err.matrix$elnet.train[i] = rsqu.train.elnet
  err.matrix$elnet.test[i] = rsqu.test.elnet
  
  end = Sys.time()
  tcv.elnet = endcv - start
  t.elnet = end - start
  
  print("Elnet done ")
  
  #LASSO
  start = Sys.time()
  
  cv.lasso = cv.glmnet(x.train, y.train, alpha = 1, lambda = lam.las, nfolds = 10)
  endcv = Sys.time()
  
  fit.lasso = glmnet(x.train, y.train, alpha = 1, lambda = cv.lasso$lambda.min)
  yhat.train.lasso = predict(fit.lasso, newx = x.train, type = "response") 
  yhat.test.lasso = predict (fit.lasso, newx = x.test, type = "response")
  
  rsqu.train.lasso = 1 - (mean((y.train - yhat.train.lasso)^2) / mean((y.train - mean(y.train))^2))
  rsqu.test.lasso = 1 - (mean((y.test - yhat.test.lasso)^2) / mean((y.test - mean(y.test))^2))

  err.matrix$lasso.train[i] = rsqu.train.lasso
  err.matrix$lasso.test[i] = rsqu.test.lasso
  
  end = Sys.time()
  tcv.lasso = endcv - start
  t.lasso = end - start
  
  print("Lasso done ")
  
  # RANDOM FOREST
  start = Sys.time()
  
  rf = randomForest(Passes.per.match~.,data = train, importance = TRUE, type = "gaussian")
  endcv = Sys.time()
  
  yhat.train.rf = predict(rf, newdata = train)
  yhat.test.rf = predict(rf, newdata = test)
  
  rsqu.train.rf = 1 - (mean((y.train - yhat.train.rf)^2) / mean((y.train - mean(y.train))^2))
  rsqu.test.rf = 1 - (mean((y.test - yhat.test.rf)^2) / mean((y.test - mean(y.test))^2))
  
  err.matrix$rf.train[i] = rsqu.train.rf
  err.matrix$rf.test[i] = rsqu.test.rf
  
  end = Sys.time()
  tcv.rf = endcv - start
  t.rf = end - start
  
  print("rf done")
  
  print(i)
  
  }

#TIME STAMPS

print("Ridge CV / TOTAL Time")
tcv.ridge
t.ridge

print("Elnet CV / TOTAL Time")
tcv.elnet
t.elnet

print("Lasso CV / TOTAL Time")
tcv.lasso
t.lasso

print("RF Time FIT/TOTAL")
tcv.rf
t.rf

#T-TEST 
t.test(err.matrix$ridge.test, conf.level= .90)
t.test(err.matrix$elnet.test, conf.level = .90)
t.test(err.matrix$lasso.test, conf.level = .90)
t.test(err.matrix$rf.test, conf.level = .90)

# RSQU PLOTS
par(mfcol = c(1, 2))

#R-square train
temp_train = cbind(err.matrix$ridge.train,err.matrix$elnet.train,
                  err.matrix$lasso.train, err.matrix$rf.train)
colnames(temp_train) = (c("Ridge", "Elnet", "Lasso", "RF"))
boxplot(temp_train, ylim=c(ymin = 0, ymax = 1),  xlab = "Training Set", ylab = "R-Squ", main = "Training Set R-Sq")

#R-square train
temp_test = cbind(err.matrix$ridge.test,err.matrix$elnet.test,
                  err.matrix$lasso.test, err.matrix$rf.test)
colnames(temp_test) = (c("Ridge", "Elnet", "Lasso", "RF"))
boxplot(temp_test, ylim=c(ymin = 0, ymax = 1), xlab = "Test Set", ylab = "R-Squ", main = "Test Set R-Sq")


#Residual plots
ridge_r = y.train - yhat.train.ridge
elnet_r = y.train - yhat.train.elnet
lasso_r = y.train - yhat.train.lasso
rf_r = y.train - yhat.train.rf

ridge_rt = y.test - yhat.test.ridge
elnet_rt = y.test - yhat.test.elnet
lasso_rt = y.test - yhat.test.lasso
rf_rt = y.test - yhat.test.rf

# Residual plots
temp_r_train = cbind(ridge_r, elnet_r, lasso_r, rf_r)
colnames(temp_r_train) = (c("Ridge", "Elnet", "Lasso", "RF"))
temp_r_test = cbind(ridge_rt, elnet_rt, lasso_rt, rf_rt)
colnames(temp_r_test) = (c("Ridge", "Elnet", "Lasso", "RF"))

par(mfcol = c(1, 2))
boxplot(temp_r_train, xlab = "Training Set", ylab = "Residuals", main = "Train Residuals",ylim = (c(-40,80)))
boxplot(temp_r_test, xlab = "Test Set", ylab = "Residuals", main = "Test Residuals", ylim = (c(-40,80)))



# PLOT CV CURVES

par(mfcol = c(1, 3))
par(mar=c(5,3,5,2)+0.1)
plot(cv.ridge,ylim = c(ymin = 140, ymax = 230))
title(main= "Ridge")
plot(cv.elnet, ylim = c(ymin = 140, ymax = 230))
title(main= "Elnet")
plot(cv.lasso, ylim = c(ymin = 140, ymax = 230))
title(main= "Lasso")

# VARIABLE IMPORTANCE 

# pull variable names from elnet, put into ordered list
var.names = rownames(fit.elnet$beta["Dimnames" = TRUE])

#standardize coefficients
S = apply(subset(soccer, select = -Passes.per.match), 2, sd)
  
# Order variables per Elnet importance
elnet.importance = rev(order(abs(fit.elnet$beta*S)))

# create a matrix to store ordered elnet variables & associated values for other models
betas.matrix = as.data.frame(matrix(nrow = length(fit.elnet$beta)))


for (i in 1:length(fit.elnet$beta)){
  var.index = elnet.importance[i]
  print(var.index)
  betas.matrix$Variable[i] = var.names[var.index]
  betas.matrix$elnet[i] = fit.elnet$beta[var.index]*S[var.index]
  betas.matrix$ridge[i] = fit.ridge$beta[var.index]*S[var.index] 
  betas.matrix$lasso[i] = fit.lasso$beta[var.index]*S[var.index]
  betas.matrix$random.forest[i]  = rf$importance[var.index]*S[var.index]
}


# Beta Plots 
par(mfcol = c(4, 1))
par(mar=c(2,3,3,1))
barplot(betas.matrix$ridge, names.arg = NULL, xlab = NULL, main = "Ridge Betas")
barplot(betas.matrix$elnet, names.arg = NULL, xlab = NULL, main = "Elnet Betas")
barplot(betas.matrix$lasso, names.arg = NULL, xlab = NULL, main = "Lasso Betas")
par(mar=c(5,3,3,1))
barplot(betas.matrix$random.forest, names.arg = betas.matrix$Variable, las = 3, main = "RF-Importance", cex.lab = 0.8)


