# Import necessary libraries
library(readr)
library(keras)
library(reticulate)
pd <- import("pandas")
# Run if you get an error: reticulate::py_last_error()

# Load training data sets
X_train <- gene_exp_data
Y_train <- prism_data_new
x_train <- ccle_data

# Load Pre-Trained Weights
tcga_geneExp_weights <- pd$read_pickle("Data/gene_DepMap_21Q4/tcga_pretrained_autoencoder_geneExp.pickle")
tcga_geneExp_model <- load_model_hdf5("Data/gene_DepMap_21Q4/tcga_pretrained_autoencoder_geneExp_model.h5")


# Make training data sets into matrices
X_train <- data.matrix(X_train, rownames.force=NA)
Y_train <- data.matrix(Y_train, rownames.force=NA)
x_train <- data.matrix(x_train, rownames.force=NA)

# Build a sequential model
model <- keras_model_sequential()
model %>%
  layer_dense(units=1024, activation='relu', input_shape=c(15363), name='input_layer', weights=tcga_geneExp_weights[[1]]) %>%
  layer_activation_leaky_relu(weights=tcga_geneExp_weights[[2]], name='feature_extractorE_layer1') %>%
  layer_dense(256, weights=tcga_geneExp_weights[[3]], name='feature_extractorE_layer2') %>%
  layer_activation_leaky_relu(weights=tcga_geneExp_weights[[4]], name='feature_extractorE_layer3') %>%
  layer_dense(64, weights=tcga_geneExp_weights[[5]], name='feature_extractorE_layer4') %>%
  layer_activation_leaky_relu(weights=tcga_geneExp_weights[[6]], name='feature_extractorE_layer5') %>%
  layer_dense(units=4686, activation='relu', name='output_layer')

# Freezing the weights of the first two layers of the model for feature extraction
freeze_weights(model, from="feature_extractorE_layer1", to="feature_extractorE_layer2")

# Summary of the frozen-model's structure
summary(model)

# Compile the frozen-model using adam and measuring loss using mse
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate=0.001)
)

# Fit the frozen-model
history <- model %>% fit(x_train, Y_train, epochs=30, batch_size=128, validation_split=0.2)

# Plot the loss of the frozen-model
plot(history)

# Unfreeze the weights of the first two layers of the frozen-model for fine-tuning
unfreeze_weights(model, from="input_layer", to="output_layer")

# Summary of the model with the first two layers having been unfrozen
summary(model)

# Compile the unfrozen model with adam optimizer and measuring loss with mse
model %>% compile(
  loss = 'mean_squared_error',
  optimizer= optimizer_adam(learning_rate=0.001)
)

# Fit the unfrozen model with fine tuning epochs of 30
history <- model %>% fit(x_train, Y_train, epochs=30, batch_size=128, validation_split=0.2)

# Plot the unfrozen-model's loss
plot(history)

# Save model in Model Data
save_model_hdf5(model, 'Model Data/model.h5', overwrite = TRUE, include_optimizer = TRUE)

# Uncomment to load saved model
#loaded_model <- load_model_hdf5("Model Data/model.h5")