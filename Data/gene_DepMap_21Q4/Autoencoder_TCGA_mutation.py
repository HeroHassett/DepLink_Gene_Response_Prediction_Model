from __future__ import print_function, division
import pandas as pd
import keras
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU, ReLU
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding, Permute, Dropout, BatchNormalization

# Load Data, TCGA gene expression data (9059,15363)
data_tcga_mutation, data_labels_tcga_mutation, sample_names_tcga_mutation, gene_names_tcga_mutation = pd.read_csv("old/tcga_mut_data.txt")
tumor_name_samples = pd.read_csv("tcga_sample_tumor.csv")
tumor_name_samples = np.asarray(tumor_name_samples[3])

# id_rand = np.random.RandomState(seed=0).permutation(data_tcga_mutation.shape[0])
X_tcga = data_tcga_mutation

X_train, X_test, y_train, y_test = train_test_split(X_tcga, tumor_name_samples, test_size=0.10, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=51)

# Set the model parameters
init = 'he_uniform'
batch_size = 64
num_epoch = 50

# save model parameters to pickle
def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)


autoencoder_model = Sequential()
autoencoder_model.add(Dense(1024, input_shape=(18281,)))
autoencoder_model.add(ReLU())
autoencoder_model.add(Dense(256))
autoencoder_model.add(ReLU())
autoencoder_model.add(Dense(64))
autoencoder_model.add(ReLU())
autoencoder_model.add(Dense(256))
autoencoder_model.add(ReLU())
autoencoder_model.add(Dense(1024))
autoencoder_model.add(ReLU())
autoencoder_model.add(Dense(18281))
autoencoder_model.add(ReLU())

autoencoder_model.summary()
callbacks = keras.callbacks.EarlyStopping(monitor="loss", patience=3)
# opt = tf.keras.optimizers.Adam(lr=0.00005)
autoencoder_model.compile(optimizer="adam", loss='mse')
history = autoencoder_model.fit(X_tcga, X_tcga, epochs=num_epoch, batch_size=batch_size, shuffle=True,
                                validation_split=0.05)

save_weight_to_pickle(autoencoder_model, 'tcga_pretrained_autoencoder_geneMutation.pickle')
autoencoder_model.save('tcga_pretrained_autoencoder_geneMutation.h5')

# print loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training loss \n 1024-256-64-batch64-lossAdam-epoch50')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["Train", "Validation"])
plt.savefig("Autoencoder_mutation_loss.png")
plt.show()

# Test dataset
X_test_decoded = autoencoder_model.evaluate(X_test, X_test)

i = 1