install.packages("googledrive")
library(googledrive)
drive_auth()
# Specify the file path
file_path <- "Machine Learning with R Final Project.R"

# Upload to Google Drive
drive_upload(file_path, path = "Machine Learning Final Project")  # Optional: Specify the folder in Google Drive

#Machine Learning with R Final Project
#Microsoft Stock Prices 2019-2024
#Time Series
# Load necessary libraries
library(xts)
library(forecast)
library(ggplot2)
library(zoo)
library(tseries)  # For ADF test
# Import data
library(readr)
data <- read_csv("Microsoft Stock Price October 2019 - October 2024.csv")
# Ensure dates are sorted and convert Date column to Date format
data$Date <- as.Date(data$Date, format="%b %d, %Y")
data <- data[order(data$Date), ]
# Handle missing values in 'Close' column with linear interpolation
data$Close <- na.approx(data$Close)
# Plot the stock closing prices over time
ggplot(data, aes(x=Date, y=Close)) + 
  geom_line() + 
  labs(title="Microsoft Stock Price (Oct 2019 - Oct 2024)", 
       x="Date", 
       y="Close Price") + 
  theme_minimal()
# Convert 'Close' column into a time series object
stock_ts <- ts(data$Close, frequency = 252)  # Assuming 252 trading days in a year
# Decompose the time series
decomposed <- decompose(stock_ts)
plot(decomposed)
# Perform the ADF test to check stationarity
adf_result <- adf.test(stock_ts)
print(adf_result)
stock_ts_diff <- diff(stock_ts)
# Split data into training and testing sets (80-20 split)
train_size <- floor(0.8 * length(stock_ts))
train <- window(stock_ts, end = c(1, train_size))  # Training set
test <- window(stock_ts, start = c(1, train_size + 1))  # Testing set
# Fit ARIMA model on training data
arima_model <- auto.arima(train)
summary(arima_model)
# Make predictions on the test data
predictions <- forecast(arima_model, h = length(test))
# Plot actual vs predicted values
plot(predictions, main = "ARIMA Forecast vs Actual")
lines(test, col = "red", lwd = 2)  # Overlay actual test data
legend("topleft", legend = c("Forecast", "Actual"), col = c("blue", "red"), lty = 1, lwd = 2)
# Check residuals of the ARIMA model
checkresiduals(arima_model)


#Regression Models
# Load necessary libraries
library(caret)      # For data splitting and evaluation
library(dplyr)      # For data manipulation
library(neuralnet)  # For deep learning models

# Prepare lagged features (e.g., using previous 5 days' closing prices as predictors)
data$Lag1 <- lag(data$Close, 1)
data$Lag2 <- lag(data$Close, 2)
data$Lag3 <- lag(data$Close, 3)
data$Lag4 <- lag(data$Close, 4)
data$Lag5 <- lag(data$Close, 5)
data$SMA5 <- rollmean(data$Close, k = 5, fill = NA)
# Remove NA rows created due to lagging
data_lagged <- na.omit(data)
# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data_lagged$Close, p = 0.8, list = FALSE)
train_data <- data_lagged[train_index, ]
test_data <- data_lagged[-train_index, ]
# Fit Linear Regression Model
lm_model <- lm(Close ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5, data = train_data)
# Summary of the model
summary(lm_model)
# Predict on the test set
lm_predictions <- predict(lm_model, newdata = test_data)
# Evaluate the model
lm_rmse <- sqrt(mean((test_data$Close - lm_predictions)^2))
print(paste("Linear Regression RMSE:", lm_rmse))
# Plot actual vs predicted
plot(test_data$Close, type = "l", col = "blue", lwd = 2, main = "Linear Regression Predictions")
lines(lm_predictions, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)
library(glmnet)
# Prepare data for glmnet (requires matrix input)
train_matrix <- as.matrix(train_data[, -1])  # Exclude the dependent variable
test_matrix <- as.matrix(test_data[, -1])    # Exclude the dependent variable
train_target <- train_data$Close             # Dependent variable
# Perform cross-validation to find the optimal lambda
cv_ridge <- cv.glmnet(train_matrix, train_target, alpha=0)
# Get the optimal lambda
optimal_lambda <- cv_ridge$lambda.min
# Fit Ridge regression model using the optimal lambda
ridge_model <- glmnet(train_matrix, train_target, alpha=0, lambda=optimal_lambda)
# Predict on test data
ridge_pred <- predict(ridge_model, newx=test_matrix)
# Calculate RMSE for Ridge Regression
ridge_rmse <- sqrt(mean((test_data$Close - ridge_pred)^2))
print(paste("Ridge Regression RMSE:", ridge_rmse))
# Plot actual vs predicted values for Ridge Regression
plot(test_data$Close, type = "l", col = "blue", lwd = 2, main = "Ridge Regression: Actual vs Predicted")
lines(ridge_pred, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Ridge Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)
# Perform cross-validation to find the optimal lambda for Lasso
cv_lasso <- cv.glmnet(train_matrix, train_target, alpha=1)  # Lasso (alpha=1)
best_lambda <- cv_lasso$lambda.min  # Optimal λ minimizing RMSE
# Get the optimal lambda for Lasso
optimal_lambda_lasso <- cv_lasso$lambda.min
# Fit Lasso regression model using the optimal lambda
lasso_model <- glmnet(train_matrix, train_target, alpha=1, lambda=optimal_lambda_lasso)
# Predict on test data
lasso_pred <- predict(lasso_model, newx=test_matrix)
print(paste("Lasso RMSE:", sqrt(mean((test_data$Close - lasso_pred)^2))))
plot(test_data$Close, type = "l", col = "blue", lwd = 2, main = "Lasso Regression: Actual vs Predicted")
lines(lasso_pred, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Lasso Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)
# Fit Polynomial regression (degree=2)
poly_model <- lm(Close ~ poly(Date, 2), data=train_data)
poly_pred <- predict(poly_model, newdata=test_data)
# Calculate RMSE for Polynomial Regression
poly_rmse <- sqrt(mean((test_data$Close - poly_pred)^2))
print(paste("Polynomial Regression RMSE:", poly_rmse))
# Support Vector Regression
train_data <- train_data[, !(colnames(train_data) %in% c("Adj Close"))]
test_data <- test_data[, !(colnames(test_data) %in% c("Adj Close"))]
library(randomForest)
rf_model <- randomForest(Close ~ ., data=train_data)
rf_pred <- predict(rf_model, newdata=test_data)





#Deep Learning
# Scale the data for neural network training
train_scaled <- scale(train_data[, c("Close", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5")])
# Extract the scaling attributes for the lag variables
# Recalculate scaling parameters using the training data
# Correct scaling parameter extraction
scaling_params <- list(
  center = colMeans(train_data[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5")]),
  scale = apply(train_data[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5")], 2, sd)
)
scaling_params$scale["Close"] <- sd(train_data$Close)
scaling_params$center["Close"] <- mean(train_data$Close)




# Apply scaling to test data with extracted parameters
test_scaled <- scale(test_data[, c("Close", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5")],
                     center = scaling_params$center,
                     scale = scaling_params$scale)

# Define the neural network structure
nn_model <- neuralnet(Close ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5,
                      data = as.data.frame(train_scaled),
                      hidden = c(15, 10),  # More neurons in hidden layers
                      linear.output = TRUE,
                      stepmax = 1e7,       # Allow more iterations
                      threshold = 0.005)   # Set convergence threshold


# Plot the neural network structure
plot(nn_model)
# Make predictions on the test data
nn_predictions <- compute(nn_model, test_scaled)$net.result
# Rescale predictions to the original scale
nn_predictions_rescaled <- (nn_predictions * scaling_params$scale["Close"]) + scaling_params$center["Close"]
# Evaluate the neural network model
nn_rmse <- sqrt(mean((test_data$Close - nn_predictions)^2))
print(paste("Neural Network RMSE:", nn_rmse))




# Plot actual vs predicted
plot(test_data$Close, type = "l", col = "blue", lwd = 2,
     main = "Neural Network: Actual vs Predicted",
     xlab = "Index", ylab = "Close Price")
lines(nn_predictions_rescaled, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Store RMSE values for each model
model_rmse <- data.frame(
  Model = c("Linear Regression", "Ridge Regression", "Lasso Regression", "Polynomial Regression", "Random Forest", "Neural Network"),
  RMSE = c(lm_rmse, ridge_rmse, sqrt(mean((test_data$Close - lasso_pred)^2)), poly_rmse, sqrt(mean((test_data$Close - rf_pred)^2)), nn_rmse)
)

# Plot RMSE for model comparison
library(ggplot2)
ggplot(model_rmse, aes(x=Model, y=RMSE, fill=Model)) +
  geom_bar(stat="identity", show.legend = FALSE) +
  theme_minimal() +
  labs(title="Model Comparison: RMSE", y="RMSE", x="Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Set seed for reproducibility
set.seed(123)

# Convert Date column to Date type and sort
data <- data %>%
  mutate(Date = as.Date(Date, format = "%b %d, %Y")) %>%
  arrange(Date)

# Fill missing values using linear interpolation
data <- data %>%
  mutate(across(where(is.numeric), ~ na.approx(., na.rm = FALSE))) %>%
  fill(everything(), .direction = "downup")

# Calculate lagged features
data <- data %>%
  mutate(
    Lag1 = lag(Close, 1),
    Lag2 = lag(Close, 2),
    Lag3 = lag(Close, 3),
    Lag4 = lag(Close, 4),
    Lag5 = lag(Close, 5),
    SMA5 = rollmean(Close, k = 5, fill = NA, align = "right"),
    SMA10 = rollmean(Close, k = 10, fill = NA, align = "right"),
    Volatility = rollapply(Close, width = 5, FUN = sd, fill = NA, align = "right")
  ) %>%
  drop_na()

# Split data into training (80%) and testing (20%) sets
train_size <- floor(0.8 * nrow(data))
train_data <- data[1:train_size, ]
test_data <- data[(train_size + 1):nrow(data), ]

# Prepare matrices for glmnet
train_matrix <- as.matrix(train_data %>% select(Lag1, Lag2, Lag3, Lag4, Lag5, SMA5, SMA10, Volatility))
test_matrix <- as.matrix(test_data %>% select(Lag1, Lag2, Lag3, Lag4, Lag5, SMA5, SMA10, Volatility))
train_target <- train_data$Close

# Train Lasso regression model with cross-validation to find optimal lambda
cv_lasso <- cv.glmnet(train_matrix, train_target, alpha = 1)
optimal_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(train_matrix, train_target, alpha = 1, lambda = optimal_lambda)

# Predict on test data
lasso_test_predictions <- predict(lasso_model, newx = test_matrix)

# Calculate RMSE for test data
lasso_test_rmse <- sqrt(mean((test_data$Close - lasso_test_predictions)^2))
print(paste("Lasso Regression Test RMSE:", lasso_test_rmse))

# Plot actual vs predicted for test data
ggplot() +
  geom_line(aes(x = test_data$Date, y = test_data$Close, color = "Actual")) +
  geom_line(aes(x = test_data$Date, y = lasso_test_predictions, color = "Predicted")) +
  labs(title = "Lasso Regression: Actual vs Predicted on Test Data",
       x = "Date", y = "Close Price") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Generate future dates for 2025 (252 trading days)
future_dates <- seq.Date(from = max(data$Date) + 1, by = "day", length.out = 365)
future_dates <- future_dates[weekdays(future_dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
future_dates <- future_dates[1:252]

# Initialize future data frame
future_data <- data.frame(Date = future_dates)
future_data <- future_data %>%
  mutate(
    Lag1 = NA, Lag2 = NA, Lag3 = NA, Lag4 = NA, Lag5 = NA,
    SMA5 = NA, SMA10 = NA, Volatility = NA, Close = NA
  )
future_data$Open <- rep(mean(train_data$Open, na.rm = TRUE), nrow(future_data))
future_data$High <- rep(mean(train_data$High, na.rm = TRUE), nrow(future_data))
future_data$Low <- rep(mean(train_data$Low, na.rm = TRUE), nrow(future_data))
future_data$`Adj Close` <- rep(mean(train_data$`Adj Close`, na.rm = TRUE), nrow(future_data))
future_data$Volume <- rep(mean(train_data$Volume, na.rm = TRUE), nrow(future_data))
future_data$SMA5[1] <- tail(train_data$SMA5, 1)
future_data$SMA10[1] <- tail(train_data$SMA10, 1)
future_data$Volatility[1] <- tail(train_data$Volatility, 1)
future_data$SMA5 <- zoo::na.locf(future_data$SMA5, na.rm = FALSE)
future_data$SMA10 <- zoo::na.locf(future_data$SMA10, na.rm = FALSE)
future_data$Volatility <- zoo::na.locf(future_data$Volatility, na.rm = FALSE)

# Set initial lags based on the last available data
last_known_close <- tail(data$Close, 1)
future_data$Lag1[1] <- last_known_close
future_data$Lag2[1] <- tail(data$Lag1, 1)
future_data$Lag3[1] <- tail(data$Lag2, 1)
future_data$Lag4[1] <- tail(data$Lag3, 1)
future_data$Lag5[1] <- tail(data$Lag4, 1)

# Predict future prices iteratively
for (i in 1:nrow(future_data)) {
  # Update lagged features
  if (i > 1) {
    future_data$Lag1[i] <- future_data$Close[i - 1]
    future_data$Lag2[i] <- ifelse(i > 2, future_data$Close[i - 2], future_data$Lag1[i])
    future_data$Lag3[i] <- ifelse(i > 3, future_data$Close[i - 3], future_data$Lag2[i])
    future_data$Lag4[i] <- ifelse(i > 4, future_data$Close[i - 4], future_data$Lag3[i])
    future_data$Lag5[i] <- ifelse(i > 5, future_data$Close[i - 5], future_data$Lag4[i])
  }
  
  # Update rolling features
  rolling_close <- c(tail(data$Close, 5), future_data$Close[1:(i - 1)])
  future_data$SMA5[i] <- ifelse(length(rolling_close) >= 5, mean(tail(rolling_close, 5)), NA)
  future_data$SMA10[i] <- ifelse(length(rolling_close) >= 10, mean(tail(rolling_close, 10)), NA)
  future_data$Volatility[i] <- ifelse(length(rolling_close) >= 5, sd(tail(rolling_close, 5)), NA)
  
  # Prepare row for prediction
  row <- future_data[i, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "SMA5", "SMA10", "Volatility"), drop = FALSE]
  row[is.na(row)] <- 0  # Replace NAs with zeros
  
  # Predict Close for this day
  future_data$Close[i] <- predict(lasso_model, newx = as.matrix(row))
}

# Plot the lasso regression prediction
# Enhanced Plot with Trend Line and Key Points
ggplot(future_data, aes(x = Date, y = Close)) +
  geom_line(color = "green", size = 1) +  # Main line
  labs(title = "Lasso Regression Prediction 2025",
       x = "Date",
       y = "Predicted Close Price") +
  theme_minimal()

library(ggplot2)

# Function to generate prediction plot
generate_prediction_plot <- function(model_name, predictions, title_color = "blue") {
  ggplot(future_data, aes(x = Date, y = predictions)) +
    geom_line(color = title_color, size = 1) +
    labs(
      title = paste(model_name, "Prediction for 2025"),
      x = "Date",
      y = "Predicted Close Price"
    ) +
    theme_minimal()
}

# Linear Regression Prediction Plot
linear_predictions <- predict(lm_model, newdata = future_data)
future_data$Linear <- linear_predictions

ggplot(future_data, aes(x = Date, y = Linear)) +
  geom_line(color = "blue", size = 1) +
  labs(
    title = "Linear Regression Prediction for 2025",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal()

# Ridge Regression Prediction Plot
ridge_coefficients <- coef(ridge_model)
print(ridge_coefficients)
print(length(ridge_coefficients))  # Should match the 12 features reported

ridge_features <- rownames(ridge_coefficients)[-1]  # Exclude intercept
print(ridge_features)
print(length(ridge_features))

row <- as.data.frame(future_data_subset[i, , drop = FALSE])
row[is.na(row)] <- 0  # Replace NA with 0
missing_features <- setdiff(ridge_features, colnames(row))
for (feature in missing_features) {
  row[[feature]] <- 0  # Add missing features with default value
}
row <- row[, ridge_features, drop = FALSE]  # Align columns with ridge_features



print(colnames(row))  # Should match ridge_features
print(length(colnames(row)))  # Should match the expected 12


future_data_subset <- future_data[, colnames(train_matrix), drop = FALSE]
future_data_subset[is.na(future_data_subset)] <- 0

future_predictions$Ridge <- rep(NA, nrow(future_data_subset))
for (i in 1:nrow(future_data_subset)) {
  row <- as.data.frame(future_data_subset[i, , drop = FALSE])
  row[is.na(row)] <- 0
  missing_features <- setdiff(ridge_features, colnames(row))
  for (feature in missing_features) {
    row[[feature]] <- 0
  }
  row <- row[, ridge_features, drop = FALSE]
  future_predictions$Ridge[i] <- predict(ridge_model, newx = as.matrix(row))
}

missing_features <- setdiff(colnames(train_matrix), colnames(row))
for (feature in missing_features) {
  row[[feature]] <- 0  # Default value for missing features
}
row <- row[, colnames(train_matrix), drop = FALSE]  # Align columns


ridge_predictions <- future_predictions$Ridge  # If computed
ggplot(future_data, aes(x = Date, y = ridge_predictions)) +
  geom_line(color = "red", size = 1) +
  labs(
    title = "Ridge Regression Prediction for 2025",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal()
future_data$Ridge <- future_predictions$Ridge  # Assuming future_predictions$Ridge exists

# Polynomial Regression Prediction Plot
polynomial_predictions <- predict(poly_model, newdata = future_data)
future_data$Polynomial <- polynomial_predictions

ggplot(future_data, aes(x = Date, y = Polynomial)) +
  geom_line(color = "green", size = 1) +
  labs(
    title = "Polynomial Regression Prediction for 2025",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal()

# Random Forest Prediction Plot
required_features <- colnames(train_data)
# Add missing features to future_data
missing_features <- c("Open", "High", "Low", "Adj Close", "Volume")

for (feature in missing_features) {
  if (!feature %in% colnames(future_data)) {
    # Use the last known value from the original dataset
    future_data[[feature]] <- tail(data[[feature]], 1)
  }
}

missing_features <- setdiff(colnames(train_data), colnames(future_data))
print(paste("Missing features:", paste(missing_features, collapse = ", ")))

if (length(missing_features) > 0) {
  print(paste("Missing features in future_data:", paste(missing_features, collapse = ", ")))
  # Add missing features with default or placeholder values
  for (feature in missing_features) {
    future_data[[feature]] <- NA  # Use NA or another sensible default
  }
}

random_forest_predictions <- predict(rf_model, newdata = future_data)
future_data$RandomForest <- random_forest_predictions

ggplot(future_data, aes(x = Date, y = RandomForest)) +
  geom_line(color = "orange", size = 1) +
  labs(
    title = "Random Forest Prediction for 2025",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal()

library(ggplot2)

ggplot(future_data, aes(x = Date, y = RandomForest)) +
  geom_line(color = "orange", size = 1) +
  labs(
    title = "Random Forest Prediction for Microsoft Stock Prices (2025)",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal() +
  geom_point(data = future_data[seq(1, nrow(future_data), by = 50), ], 
             aes(x = Date, y = RandomForest), color = "blue", size = 2) +
  annotate("text", x = future_data$Date[50], y = future_data$RandomForest[50] + 5,
           label = "Stabilization phase", color = "blue", size = 3.5) +
  annotate("text", x = future_data$Date[200], y = future_data$RandomForest[200] - 5,
           label = "Sharp increase", color = "red", size = 3.5) +
  annotate("rect", xmin = future_data$Date[150], xmax = future_data$Date[180],
           ymin = min(future_data$RandomForest), ymax = max(future_data$RandomForest),
           fill = "yellow", alpha = 0.2) +
  annotate("text", x = future_data$Date[165], y = max(future_data$RandomForest) + 10,
           label = "Volatile phase", color = "purple", size = 3.5)


# Neural Network Prediction Plot
# Check if scaling parameters exist and are correctly aligned
if (length(scaling_params$center) != 5 || length(scaling_params$scale) != 5) {
  stop("Scaling parameters are not correctly aligned with the neural network input features.")
}

# Scale future_data for neural network prediction
nn_scaled_future <- scale(future_data[, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5")], 
                          center = scaling_params$center, 
                          scale = scaling_params$scale)

# Generate predictions using the neural network
neural_network_predictions <- compute(nn_model, nn_scaled_future)$net.result
test_rescaled <- neural_network_predictions[1:5] * as.numeric(scaling_params$scale["Close"]) +
  as.numeric(scaling_params$center["Close"])
print(test_rescaled)

# Rescale predictions back to original scale
neural_network_rescaled <- neural_network_predictions * as.numeric(scaling_params$scale["Close"]) +
  as.numeric(scaling_params$center["Close"])

# Add predictions to future_data
future_data$NeuralNetwork <- ifelse(is.na(neural_network_rescaled) | !is.finite(neural_network_rescaled), 0, neural_network_rescaled)
future_data$NeuralNetwork <- neural_network_rescaled

# Plot Neural Network Prediction
ggplot(future_data, aes(x = Date, y = NeuralNetwork)) +
  geom_line(color = "purple", size = 1) +
  labs(
    title = "Neural Network Prediction for 2025",
    x = "Date",
    y = "Predicted Close Price"
  ) +
  theme_minimal()
# Add Lasso Regression predictions to future_data
future_data$Lasso <- predict(lasso_model, newx = as.matrix(future_data[, colnames(train_matrix), drop = FALSE]))

# Ensure all necessary columns are present
comparison_plot <- ggplot() +
  geom_line(data = future_data, aes(x = Date, y = Linear, color = "Linear Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Ridge, color = "Ridge Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Polynomial, color = "Polynomial Regression")) +
  geom_line(data = future_data, aes(x = Date, y = RandomForest, color = "Random Forest")) +
  geom_line(data = future_data, aes(x = Date, y = NeuralNetwork, color = "Neural Network")) +
  geom_line(data = future_data, aes(x = Date, y = Lasso, color = "Lasso Regression")) +
  labs(
    title = "Microsoft Stock Price Prediction for 2025 (Model Comparison)",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  ) +
  theme_minimal()

# Display the plot
print(comparison_plot)


#Overlaying Predictions on One Graph
ggplot() +
  geom_line(data = future_data, aes(x = Date, y = Linear, color = "Linear Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Ridge, color = "Ridge Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Polynomial, color = "Polynomial Regression")) +
  geom_line(data = future_data, aes(x = Date, y = RandomForest, color = "Random Forest")) +
  geom_line(data = future_data, aes(x = Date, y = NeuralNetwork, color = "Neural Network")) +
  geom_line(data = future_data, aes(x = Date, y = Lasso, color = "Lasso Regression")) + 
  labs(
    title = "Microsoft Stock Price Prediction for 2025 (Model Comparison)",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  ) +
  theme_minimal()

library(ggplot2)

# Generate combined prediction plot with annotations
ggplot() +
  geom_line(data = future_data, aes(x = Date, y = Linear, color = "Linear Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Ridge, color = "Ridge Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Polynomial, color = "Polynomial Regression")) +
  geom_line(data = future_data, aes(x = Date, y = RandomForest, color = "Random Forest")) +
  geom_line(data = future_data, aes(x = Date, y = NeuralNetwork, color = "Neural Network")) +
  geom_line(data = future_data, aes(x = Date, y = Lasso, color = "Lasso Regression")) +
  labs(
    title = "Model Predictions for Microsoft Stock Prices (2025)",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  ) +
  theme_minimal() +
  # Alignment Annotation
  annotate("text", x = future_data$Date[100], y = mean(c(
    future_data$Ridge[100], future_data$RandomForest[100], future_data$NeuralNetwork[100])) + 5,
    label = "Convergent trends (High confidence)", color = "darkgreen", size = 3.5) +
  geom_vline(xintercept = future_data$Date[100], linetype = "dotted", color = "darkgreen") +
  
  # Divergence Annotation
  annotate("text", x = future_data$Date[160], y = max(future_data$NeuralNetwork[160:170]) + 10,
           label = "Divergent trends (High uncertainty)", color = "red", size = 3.5) +
  annotate("segment", x = future_data$Date[150], xend = future_data$Date[170],
           y = min(c(future_data$Linear[150:170], future_data$NeuralNetwork[150:170])),
           yend = max(c(future_data$Linear[150:170], future_data$NeuralNetwork[150:170])),
           color = "red", arrow = arrow(length = unit(0.1, "cm"))) +
  
  # Volatility Annotations
  annotate("text", x = future_data$Date[50], y = future_data$NeuralNetwork[50] + 20,
           label = "Volatile predictions (Neural Network)", color = "purple", size = 3.5) +
  annotate("text", x = future_data$Date[200], y = future_data$RandomForest[200] - 20,
           label = "Stable trend (Random Forest)", color = "orange", size = 3.5) +
  
  # Add transparent regions for clarity in volatility or convergence
  annotate("rect", xmin = future_data$Date[90], xmax = future_data$Date[110],
           ymin = min(future_data$RandomForest[90:110]),
           ymax = max(future_data$RandomForest[90:110]),
           alpha = 0.2, fill = "green") +
  # Highlight trend consistency
  annotate("text", x = future_data$Date[100], y = future_data$RandomForest[100] + 10,
           label = "Stable trend (Random Forest)", color = "orange", size = 3.5) +
  annotate("text", x = future_data$Date[50], y = future_data$Linear[50] - 15,
           label = "Downward trend (Linear)", color = "blue", size = 3.5) +
  
  # Emphasize outliers and anomalies
  geom_point(data = future_data[future_data$NeuralNetwork > max(future_data$RandomForest) + 10, ],
             aes(x = Date, y = NeuralNetwork), color = "purple", size = 2) +
  annotate("text", x = future_data$Date[200], y = max(future_data$NeuralNetwork) + 10,
           label = "Anomalous spike (Neural Network)", color = "purple", size = 3.5) +
  
  # Show variance among predictions
  annotate("rect", xmin = future_data$Date[150], xmax = future_data$Date[180],
           ymin = min(c(future_data$Linear, future_data$RandomForest)),
           ymax = max(c(future_data$NeuralNetwork, future_data$Polynomial)),
           fill = "yellow", alpha = 0.2) +
  annotate("text", x = future_data$Date[165], y = max(future_data$NeuralNetwork) + 20,
           label = "High variance among models", color = "red", size = 3.5) +
  
  # Legend and styling
  theme(legend.position = "bottom") +
  scale_color_manual(values = c(
    "Linear Regression" = "blue",
    "Ridge Regression" = "red",
    "Polynomial Regression" = "green",
    "Random Forest" = "orange",
    "Neural Network" = "purple",
    "Lasso Regression" = "black"
  ))

# Add Lasso Regression predictions to future_data
future_data$Lasso <- predict(lasso_model, newx = as.matrix(future_data[, colnames(train_matrix), drop = FALSE]))
ggplot() + 
  geom_line(data = future_data, aes(x = Date, y = Lasso, color = "Lasso Regression")) + 
  labs(
    title = "Microsoft Stock Prediction 2025 Lasso Regression",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  )
# Ensure all necessary columns are present
comparison_plot <- ggplot() +
  geom_line(data = future_data, aes(x = Date, y = Linear, color = "Linear Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Ridge, color = "Ridge Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Polynomial, color = "Polynomial Regression")) +
  geom_line(data = future_data, aes(x = Date, y = RandomForest, color = "Random Forest")) +
  geom_line(data = future_data, aes(x = Date, y = NeuralNetwork, color = "Neural Network")) +
  geom_line(data = future_data, aes(x = Date, y = Lasso, color = "Lasso Regression")) +
  labs(
    title = "Microsoft Stock Price Prediction for 2025 (Model Comparison)",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  ) +
  theme_minimal()

# Display the plot
print(comparison_plot)

# Generate combined prediction plot with advanced annotations
ggplot() +
  geom_line(data = future_data, aes(x = Date, y = Linear, color = "Linear Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Ridge, color = "Ridge Regression")) +
  geom_line(data = future_data, aes(x = Date, y = Polynomial, color = "Polynomial Regression")) +
  geom_line(data = future_data, aes(x = Date, y = RandomForest, color = "Random Forest")) +
  geom_line(data = future_data, aes(x = Date, y = NeuralNetwork, color = "Neural Network")) +
  labs(
    title = "Microsoft Stock Price Predictions (2025): Model Comparison",
    x = "Date",
    y = "Predicted Close Price",
    color = "Model"
  ) +
  theme_minimal() +
  
  # Alignment Annotation
  annotate("text", x = future_data$Date[100], y = mean(c(
    future_data$Ridge[100], future_data$RandomForest[100], future_data$NeuralNetwork[100])) + 5,
    label = "Convergent trends (High confidence)", color = "darkgreen", size = 3.5) +
  geom_vline(xintercept = future_data$Date[100], linetype = "dotted", color = "darkgreen") +
  
  # Divergence Annotation
  annotate("text", x = future_data$Date[160], y = max(future_data$NeuralNetwork[160:170]) + 10,
           label = "Divergent trends (High uncertainty)", color = "red", size = 3.5) +
  annotate("segment", x = future_data$Date[150], xend = future_data$Date[170],
           y = min(c(future_data$Linear[150:170], future_data$NeuralNetwork[150:170])),
           yend = max(c(future_data$Linear[150:170], future_data$NeuralNetwork[150:170])),
           color = "red", arrow = arrow(length = unit(0.1, "cm"))) +
  
  # Volatility Annotations
  annotate("text", x = future_data$Date[50], y = future_data$NeuralNetwork[50] + 20,
           label = "Volatile predictions (Neural Network)", color = "purple", size = 3.5) +
  annotate("text", x = future_data$Date[200], y = future_data$RandomForest[200] - 20,
           label = "Stable trend (Random Forest)", color = "orange", size = 3.5) +
  
  # Add transparent regions for clarity in volatility or convergence
  annotate("rect", xmin = future_data$Date[90], xmax = future_data$Date[110],
           ymin = min(future_data$RandomForest[90:110]),
           ymax = max(future_data$RandomForest[90:110]),
           alpha = 0.2, fill = "green") +
  
  # Customize theme
  theme(legend.position = "bottom") +
  scale_color_manual(values = c(
    "Linear Regression" = "blue",
    "Ridge Regression" = "red",
    "Polynomial Regression" = "green",
    "Random Forest" = "orange",
    "Neural Network" = "purple"
  ))

#Calculate RMSE for 2025 Prediction
model_rmse <- data.frame(
  Model = c("Linear Regression", "Ridge Regression", "Lasso Regression", "Polynomial Regression", "Random Forest", "Neural Network"),
  RMSE = c(lm_rmse, ridge_rmse, lasso_test_rmse, poly_rmse, sqrt(mean((test_data$Close - random_forest_predictions)^2)), nn_rmse)
)

print(model_rmse)

# Plot RMSE Comparison
ggplot(model_rmse, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
