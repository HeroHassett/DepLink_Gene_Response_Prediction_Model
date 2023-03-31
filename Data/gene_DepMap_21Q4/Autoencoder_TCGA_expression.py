from __future__ import print_function, division

import keras
import tensorflow
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding, Dropout, ZeroPadding2D
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras import backend as K
from functools import partial
# from keras.utils import to_categorical
from keras.models import model_from_json
# from keras.layers import D
import pandas as pd
from keras.backend import int_shape
from random import shuffle
# from sklearn.metrics import roc_auc_score
from import_data import load_data, load_geneset

import seaborn as sns
import tensorflow as tf
import pickle
# from keras import backend as k
from tensorflow.python.keras.backend import get_session
import keras.backend as k
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding, Permute, Dropout
from keras.regularizers import l1_l2
from scipy.stats import wasserstein_distance
# from keras.engine import Layer
from keras.layers import Layer

# import keras.backend as K

sns.set()
os.environ["SM_FRAMEWORK"] = "tf.keras"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# TensorFlow wizardry
config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto()

## Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True

## Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 1

# Load Data, TCGA gene expression data (9059,15363)
data_tcga_exp, data_labels_tcga_exp, sample_names_tcga_exp, gene_names_tcga_exp = load_data("tcga_exp_data.txt")
tumor_name_samples = load_data("tcga_sample_tumor.csv")
tumor_name_samples = np.asarray(tumor_name_samples[3])

id_rand = np.random.RandomState(seed=0).permutation(data_tcga_exp.shape[0])
X_tcga = data_tcga_exp[id_rand,]
# X_tcga = X_tcga[:100,:]

# Set the model parameters
activation_func = 'relu'
activation_func2 = 'linear'
init = 'he_uniform'
dense_layer_dim = 128
batch_size = 128
num_epoch = 25
BATCH_SIZE = 16
Number_epochs = 20000


# save model parameters to pickle
def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)


from tensorflow.keras import layers
autoencoder_model = Sequential()
autoencoder_model.add(Dense(1024, input_shape=(15363,)))
autoencoder_model.add(LeakyReLU())
autoencoder_model.add(Dense(256))
autoencoder_model.add(LeakyReLU())
autoencoder_model.add(Dense(64))
autoencoder_model.add(LeakyReLU())
autoencoder_model.add(Dense(256))
autoencoder_model.add(LeakyReLU())
autoencoder_model.add(Dense(1024))
autoencoder_model.add(LeakyReLU())
autoencoder_model.add(Dense(15363))
# gene_exp = Input(shape=(15363,))
# pred_gene_exp = model(gene_exp)
# print(model.summary())

autoencoder_model.summary()
callbacks = keras.callbacks.EarlyStopping(monitor="loss", patience=3)
autoencoder_model.compile(optimizer="adam", loss='mean_squared_error')
history = autoencoder_model.fit(X_tcga, X_tcga, epochs=num_epoch, batch_size=batch_size, shuffle=True, validation_split=0.05,
                                callbacks=[callbacks])
# print loss
plt.plot(history.history['loss'])
plt.title('Training loss \n 1024-256-64-batch128-lossAdam-epoch100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

save_weight_to_pickle(autoencoder_model,'tcga_pretrained_autoencoder_geneExp_3.pickle')
autoencoder_model.save('tcga_pretrained_autoencoder_geneExp_model_3.h5')

# pred_X_tcga = autoencoder_model.predict(X_tcga)
# print(pred_X_tcga)


# Try UMap/TSNE use all the data from the bottelneck of encoder.
# Try the variational autoencoder and autoencoder.
# Compounded fingerprint dataset to get used to it. They are new data and this have no drugs. (*imp)
# K-fold cross validation (eg 5-Fold/10-Fold) is done for reporting. Get code ready.
# Mutation data and methylation data and think about how to combine them.
# Ask for RNA sequencing data and how to look at gene data
# R using different gene expression analysis. Can pick one website.