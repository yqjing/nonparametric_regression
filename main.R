# baisc setup
library(ISLR2) # dataset
library(glmnet) # model selection, lasso
library(mgcv) # gdm, spline
library(np) # for the kernel regression
library(DiceKriging) # gaussian process
# additional packages
library(Cairo) # for saving figures in the server
library(corrplot) # correlation matrix plot
library(dplyr) # for data manipulation
library(ggplot2) # for plotting



# underlying problem: estimate the housing price index model. Specifically, based on the current sample of houses, 
# given a new house with its 13 sociodemographic statistics, we intend to predict its price accurately, and provide a 95% credible interval for the predictions . Namely, we focus
# on the predictive modelling of the house price (Median value of owner-occupied homes in $1000's). 

# A side-focus is to find out the predictors that influence the house price most. We intend to use LASSO to attempt the
# task. 

# ====================train-test split===================
# randomly split the whole dataset into train (80%) and test (20%).
dim(Boston) # 506 13
train_idx = sample(seq_len(nrow(Boston)), size=floor(0.8 * nrow(Boston)))
train_set = Boston[train_idx, ]
test_set = Boston[-train_idx, ]
dim(train_set) # 404 13
dim(test_set) # 102 13


# ====================pre-processing=====================
head(train_set)
# model matrix & log transform the tax and lstat
X = train_set %>%
  mutate(tax = log(tax), lstat = log(lstat)) %>%
  select(-medv) %>%
  as.matrix()
# centralize and normalize the variance of the model matrix for LASSO model selection
X = scale(X, T, T)
head(X)
# response vector
y = train_set %>%
  select(medv) %>%
  as.matrix()
# display that there exist strong colinearity between predictors
corr_matrix <- cor(X, method = "spearman")
CairoPNG(filename = "correlation_heatmap.png", width = 800, height = 800)
corrplot(corr_matrix, method = "color", type = "full", tl.cex = 1.1, cl.cex = 1.1, addCoef.col = "black", number.cex = 0.8)
dev.off()
# hist for log(lstat) vs stat
CairoPNG(filename = "lstat_log_transform.png", width = 800, height = 400)
par(mfrow = c(1, 2))
hist(Boston$lstat, col = "lightblue", main = "Histogram of lstat", xlab = "lstat", freq = FALSE, breaks=30)
lines(density(Boston$lstat), col = "blue", lwd = 2)
hist(log(Boston$lstat), col = "lightgreen", main = "Histogram of log(lstat)", xlab = "log(lstat)", freq = FALSE, breaks=30)
lines(density(log(Boston$lstat)), col = "green", lwd = 2)
par(mfrow = c(1, 1))
dev.off()

# ====================model selection=====================
# LASSO 
# AIC (best subset)
# LASSO has smaller Err (expected prediction error)


# ===================model fit & prediction============================

# ====Spline_regression======
knots = 5
# fit (b-spline basis functions, additive model)
spline_reg = gam(medv ~ s(log(lstat), bs="bs", k=knots) +
                        s(rm, bs="bs", k=knots) +
                        s(ptratio, bs="bs", k=knots) +
                        s(chas, bs="bs", k=knots) + 
                        s(crim, bs="bs", k=knots) +
                        s(dis, bs="bs", k=knots) +
                        s(log(tax), bs="bs", k=knots),
                        data=train_set) 
summary(spline_reg) # k = {5: (0.85, 13.995)}
CairoPNG(filename = "linearity.png", width = 1000, height = 1000)
plot(spline_reg, pages=1, rug=T, shade=T)
dev.off()
coef(spline_reg)

# predict
y_hat = predict(spline_reg, newdata=test_set %>% select(-medv), type="link")
dim(y_hat)
head(y_hat)
# plot the y_hat against y_true in test set
results <- data.frame(Actual = test_set$medv, Predicted = y_hat)
results <- results[order(results$Actual), ]
CairoPNG(filename = "spline_predict.png", width = 1000, height = 1000)
plot(results$Actual, type = "b", col = "blue", lwd = 2, pch = 16,
     xlab = "Index", ylab = "Value", main = "Predictions vs Actual Values")
lines(results$Predicted, type = "b", col = "red", lwd = 2, pch = 16)
legend("topleft", legend = c("Actual Values", "Predicted Values"), 
       col = c("blue", "red"), lty = 1, lwd = 5, pch = 16)
dev.off()
# 95% prediction interval of spline regression assuming 
y_se <- predict(spline_reg, newdata = test_set, se.fit = TRUE, type = "link")
y_hat <- y_se$fit
se_fit <- y_se$se.fit
y_hat_upper <- y_hat + 4 * se_fit
y_hat_lower <- y_hat - 4 * se_fit
results <- data.frame(Actual = test_set$medv, Predicted = y_hat, 
                      Lower = y_hat_lower, Upper = y_hat_upper)
inside_interval <- sum(results$Actual >= results$Lower & results$Actual <= results$Upper)
total_points <- nrow(results)
percentage_inside <- (inside_interval / total_points) * 100
results <- results[order(results$Actual), ]
CairoPNG(filename = "spline_predict_band.png", width = 1000, height = 1000)
ggplot(results, aes(x = seq_along(Actual))) +
  geom_point(aes(y = Actual), color = "blue", size = 2) +
  geom_line(aes(y = Actual), color = "blue", linetype = "dotted") +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "grey", alpha = 0.5) +
  labs(x = "Index", y = "Value", title = "Predictions vs Actual Values") +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_y_continuous(limits = c(min(results$Lower), max(results$Upper)))
dev.off()
# compute the mse
mse_spline = mean((test_set$medv - y_hat)^2)
mse_spline 
# compute R squared
rss <- sum((test_set$medv - y_hat)^2)  
tss <- sum((test_set$medv - mean(test_set$medv))^2) 
r_squared_spline <- 1 - (rss / tss)
print(r_squared_spline)
# compute percent of y_true covered in credible interval
percentage_inside 




# ===kernel regression=====

# fit
train_set_1 <- train_set %>%
  mutate(tax = log(tax), lstat = log(lstat))
test_set_1 <- test_set %>%
  mutate(tax = log(tax), lstat = log(lstat))
train_features <- train_set_1 %>% select(-medv)
train_target <- train_set_1$medv
test_features <- test_set_1 %>% select(-medv)
test_target <- test_set_1$medv
scaled_train_features <- scale(train_features, center = TRUE, scale = TRUE)
train_mean <- attr(scaled_train_features, "scaled:center")
train_sd <- attr(scaled_train_features, "scaled:scale")
scaled_test_features <- scale(test_features, center = train_mean, scale = train_sd)
scaled_train_set <- data.frame(scaled_train_features, medv = train_target)
scaled_test_set <- data.frame(scaled_test_features, medv = test_target)
predictor_columns = c("crim", "chas", "rm", "dis", "tax", "ptratio", "lstat")
kernel_reg <- npreg(reformulate(predictor_columns, response = "medv"), data = scaled_train_set, regtype="ll")

# predict
predictions <- predict(kernel_reg, newdata = scaled_test_set, se.fit=T)
y_hat_kernel <- predictions$fit
se_fit_kernel <- predictions$se.fit
y_hat_upper_kernel <- y_hat_kernel + 4 * se_fit_kernel
y_hat_lower_kernel <- y_hat_kernel - 4 * se_fit_kernel
results_kernel <- data.frame(Actual = scaled_test_set$medv, Predicted = y_hat_kernel, 
                             Lower = y_hat_lower_kernel, Upper = y_hat_upper_kernel)
results_kernel <- results_kernel[order(results_kernel$Actual), ]
inside_interval_kernel <- sum(results_kernel$Actual >= results_kernel$Lower & results_kernel$Actual <= results_kernel$Upper)
total_points_kernel <- nrow(results_kernel)
percentage_inside_kernel <- (inside_interval_kernel / total_points_kernel) * 100
CairoPNG(filename = "kernel_predict_band.png", width = 1000, height = 1000)
ggplot(results_kernel, aes(x = seq_along(Actual))) +
  geom_point(aes(y = Actual), color = "blue", size = 2) +
  geom_line(aes(y = Actual), color = "blue", linetype = "dotted") +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "grey", alpha = 0.5) +
  labs(x = "Index", y = "Value", title = "Predictions vs Actual Values (Kernel Regression)") +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_y_continuous(limits = c(min(results$Lower), max(results$Upper)))
dev.off()
# compute mse
mse_kernel <- mean((scaled_test_set$medv - y_hat_kernel)^2)
print(paste("MSE for Kernel Regression:", mse_kernel))
# compute R sqaured
rss <- sum((scaled_test_set$medv - y_hat_kernel)^2)  
tss <- sum((scaled_test_set$medv - mean(scaled_test_set$medv))^2) 
r_squared_kernel <- 1 - (rss / tss)
r_squared_kernel
# compute percent
print(paste("Percentage of Actual Values within the Prediction Interval:", percentage_inside_kernel, "%"))
# CORR
corr_kernel = cor(scaled_test_set$medv, y_hat_kernel)
corr_kernel

# ===gaussian process regression (kernel method)=====
library(kernlab)
# fit
train_set_1 <- train_set %>%
  mutate(tax = log(tax), lstat = log(lstat))
test_set_1 <- test_set %>%
  mutate(tax = log(tax), lstat = log(lstat))
train_features <- train_set_1 %>% select(-medv)
train_target <- train_set_1$medv
test_features <- test_set_1 %>% select(-medv)
test_target <- test_set_1$medv
scaled_train_features <- scale(train_features, center = TRUE, scale = TRUE)
train_mean <- attr(scaled_train_features, "scaled:center")
train_sd <- attr(scaled_train_features, "scaled:scale")
scaled_test_features <- scale(test_features, center = train_mean, scale = train_sd)

scaled_train_features = scaled_train_features %>% select(crim, chas, rm, dis, tax, ptratio, lstat) %>% as.matrix()
scaled_test_features = scaled_test_features %>% select(crim, chas, rm, dis, tax, ptratio, lstat) %>% as.matrix()
train_target = as.vector(train_target)
test_target = test_target

kernel_method_mod = ksvm(scaled_train_features, train_target, kernel = "rbfdot", C = 1)
kernel_method_mod

# predict
predictions_km <- predict(kernel_method_mod, scaled_test_features)
predictions_km = predictions_km %>% as.numeric()

results_km <- data.frame(Actual = test_target, Predicted = predictions_km)
results_km <- results_km[order(results_km$Actual), ]
CairoPNG(filename = "km_predict.png", width = 1000, height = 1000)
ggplot(results_km, aes(x = seq_along(Actual))) +
  geom_point(aes(y = Actual), color = "blue", size = 2) +
  geom_line(aes(y = Actual), color = "blue", linetype = "dotted") +
  geom_point(aes(y = Predicted), color = "red", size = 2) +
  geom_line(aes(y = Predicted), color = "red", linetype = "dotted") +
  theme_minimal() +
  theme(legend.position = "top") 
dev.off()

# compute mse
mse_km <- mean((test_target - predictions_km)^2)
print(paste("MSE for Kernel method regression:", mse_kernel)) # 18.09
# compute R sqaured
rss <- sum((test_target - predictions_km)^2)  
tss <- sum((test_target - mean(test_target))^2) 
r_squared_km <- 1 - (rss / tss)
r_squared_km
# add CORR measure
corr_km = cor(test_target, predictions_km)
corr_km
# add residual plot
residuals_gp <- test_target - predictions_km
residuals_df <- data.frame(Predicted = predictions_km, Residuals = residuals_gp)
CairoPNG(filename = "km_residual_plot.png", width = 1000, height = 1000)
ggplot(residuals_df, aes(x = Predicted, y = Residuals)) +
  geom_point(color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residuals vs Predicted Values (Gaussian Process)") +
  theme_minimal()
dev.off()


