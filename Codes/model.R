# Import necessary libraries
library(readr)
library(keras)

# Load training data sets
X_train <- gene_exp_data
Y_train <- prism_data_new

# Replace row and column names with their numeric indices
#colnames(X_train) <- 1:ncol(X_train)
#colnames(Y_train) <- 1:ncol(Y_train)
#rownames(X_train) <- 1:nrow(X_train)
#rownames(Y_train) <- 1:nrow(Y_train)

# Make training data sets into matrices
X_train <- data.matrix(X_train, rownames.force=NA)
Y_train <- data.matrix(Y_train, rownames.force=NA)

# Print the dimensions of the input matrices
#print(dim(X_train))
#print(dim(Y_train))

# Build a sequential model
model <- keras_model_sequential()
model %>%
  layer_dense(units=10240, activation='relu', input_shape=c(17040), name='input_dense_layer') %>%
  layer_dense(units=4686, activation='relu', name='output_layer')

# Summary of the model's structure
summary(model)

# Compile the model using adam and measuring loss using mse
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate=0.001)
)

# Fit the model
history <- model %>% fit(X_train, Y_train, epochs=20, batch_size=128, validation_split=0.2)

# Plot the loss of the model
plot(history)

# Save model in Model Data
save_model_hdf5(model, 'Model Data/model.h5', overwrite = TRUE, include_optimizer = TRUE)

# Uncomment to load saved model
#loaded_model <- load_model_hdf5("Model Data/model.h5")