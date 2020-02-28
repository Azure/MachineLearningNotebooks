#' Modified from: "https://github.com/rstudio/keras/blob/master/vignettes/
#' examples/cifar10_cnn.R"
#' 
#' Train a simple deep CNN on the CIFAR10 small images dataset.
#'  
#' It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50
#' epochs, though it is still underfitting at that point.

library(keras)
install_keras()

library(azuremlsdk)

# Parameters --------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

batch_size <- as.numeric(args[2])
log_metric_to_run("batch_size", batch_size)

epochs <- as.numeric(args[4])
log_metric_to_run("epochs", epochs)

lr <- as.numeric(args[6])
log_metric_to_run("lr", lr)

decay <- as.numeric(args[8])
log_metric_to_run("decay", decay)

data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs  
x_train <- cifar10$train$x / 255
x_test <- cifar10$test$x / 255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)


# Defining Model ----------------------------------------------------------

# Initialize sequential model
model <- keras_model_sequential()

model %>%

# Start with hidden 2D convolutional layer being fed 32x32 pixel images
layer_conv_2d(
    filter = 32, kernel_size = c(3, 3), padding = "same",
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation("relu") %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3, 3)) %>%
  layer_activation("relu") %>%

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%

  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 32, kernel_size = c(3, 3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3, 3)) %>%
  layer_activation("relu") %>%

  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%

  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr, decay)

model %>%
    compile(loss = "categorical_crossentropy",
          optimizer = opt,
          metrics = "accuracy"
)


# Training ----------------------------------------------------------------

if (!data_augmentation) {

    model %>%
    fit(x_train,
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = list(x_test, y_test),
        shuffle = TRUE
  )

} else {

    datagen <- image_data_generator(rotation_range = 20,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  horizontal_flip = TRUE
  )

    datagen %>% fit_image_data_generator(x_train)

    results <- evaluate(model, x_train, y_train, batch_size)
    log_metric_to_run("Loss", results[[1]])
    cat("Loss: ", results[[1]], "\n")
    cat("Accuracy: ", results[[2]], "\n")
}