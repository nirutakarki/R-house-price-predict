# Load necessary libraries

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(corrplot)
library(reshape2)

# Initial Data Exploration
# Load the dataset

house_data <- read_csv("house.csv")

# Display the structure of the dataset
str(house_data)

# Display summary statistics of the dataset
summary(house_data)

# Check for missing values
colSums(is.na(house_data))

# Display the first few rows of the dataset
head(house_data)

# EDA
# Distribution of target variable (price)
ggplot(house_data, aes(x = price)) +
  geom_histogram(binwidth = 5000, fill = "skyblue", color = "red") +
  theme_minimal() +
  labs(title = "Distribution of House Prices", x = "Price", y = "Frequency")

# Scatter plots to visualize relationships between numeric variables and price
ggplot(house_data, aes(x = net_sqm, y = price)) +
  geom_point(color = "blue") +
  theme_minimal() +
  labs(title = "Net Square Meter vs Price", x = "Net Square Meter", y = "Price")

ggplot(house_data, aes(x = bedroom_count, y = price)) +
  geom_point(color = "orange") +
  theme_minimal() +
  labs(title = "Number of Bedrooms vs Price", x = "Bedrooms", y = "Price")

ggplot(house_data, aes(x = age, y = price)) +
  geom_point(color = "chocolate") +
  theme_minimal() +
  labs(title = "Age of House vs Price", x = "Age", y = "Price")

ggplot(house_data, aes(x = floor, y = price)) +
  geom_point(color = "coral2") +
  theme_minimal() +
  labs(title = "Floor vs Price", x = "Floor", y = "Price")

ggplot(house_data, aes(x = center_distance, y = price)) +
  geom_point(color = "cyan4") +
  theme_minimal() +
  labs(title = "Center Distance vs Price", x = "Center Distance", y = "Price")

ggplot(house_data, aes(x = metro_distance, y = price)) +
  geom_point(color = "deeppink1") +
  theme_minimal() +
  labs(title = "Metro Distance vs Price", x = "Metro Distance", y = "Price")

# Correlation analysis before encoding
numeric_cols_before <- house_data %>% select_if(is.numeric)
cor_matrix_before <- cor(numeric_cols_before, use = "complete.obs")

# Transform correlation matrix to long format
melted_cor_matrix_before <- melt(cor_matrix_before)

# Plot heatmap before encoding
ggplot(data = melted_cor_matrix_before, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "bisque", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Matrix Heatmap Before Encoding")

# Set seed for reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets
trainIndex <- createDataPartition(house_data$price, p = 0.8, list = FALSE)
train_data <- house_data[trainIndex, ]
test_data <- house_data[-trainIndex, ]

# Display the dimensions of the training and testing sets
dim(train_data)
dim(test_data)

# Standardize the features (excluding the target variable)
preProcValues <- preProcess(train_data, method = c("center", "scale"))

train_data_scaled <- predict(preProcValues, train_data)
test_data_scaled <- predict(preProcValues, test_data)

# Display the first few rows of the scaled training data
head(train_data_scaled)

# Train a linear regression model
linear_model <- train(price ~ ., data = train_data_scaled, method = "lm")

# Display the model summary
summary(linear_model)

# Predict on the testing set
linear_predictions <- predict(linear_model, newdata = test_data_scaled)

# Calculate performance metrics (e.g., RMSE, R-squared)
linear_performance <- postResample(linear_predictions, test_data_scaled$price)

# Train a Random Forest model
set.seed(123)
rf_model <- train(price ~ ., data = train_data_scaled, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Display the model summary
print(rf_model)

# Predict on the testing set using the Random Forest model
rf_predictions <- predict(rf_model, newdata = test_data_scaled)

# Calculate performance metrics for the Random Forest model
rf_performance <- postResample(rf_predictions, test_data_scaled$price)

# Display performance metrics for both models
model_performance <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE = c(linear_performance["RMSE"], rf_performance["RMSE"]),
  Rsquared = c(linear_performance["Rsquared"], rf_performance["Rsquared"])
)

print(model_performance)

# Plot RMSE comparison
ggplot(model_performance, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison - RMSE", x = "Model", y = "RMSE") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "none")

# Plot R-squared comparison
ggplot(model_performance, aes(x = Model, y = Rsquared, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison - R-squared", x = "Model", y = "R-squared") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "none")

