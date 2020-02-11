# This script loads a dataset of which the last column is supposed to be the
# class and logs the accuracy

library(azuremlsdk)
library(caret)
library(optparse)
library(datasets)


iris_data <- data(iris)
summary(iris_data)

in_train <- createDataPartition(y = iris_data$Species, p = .8, list = FALSE)
train_data <- iris_data[in_train,]
test_data <- iris_data[-in_train,]

# Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

set.seed(7)
model <- train(Species ~ .,
               data = train_data,
               method = "lda",
               metric = metric,
               trControl = control)
predictions <- predict(model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$Species)
message(conf_matrix)

log_metric_to_run(metric, conf_matrix$overall["Accuracy"])

saveRDS(model, file = "./outputs/model.rds")
message("Model saved")
