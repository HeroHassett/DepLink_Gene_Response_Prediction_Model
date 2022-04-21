import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from random import randint

from import_data import load_data

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.layers import Lambda, Input, Dense, Input, Flatten, Multiply, Reshape, concatenate
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Conv2DTranspose, \
    BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.callbacks import Callback

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

n_epochs = 50
klstart = 10
kl_annealtime = 20
weight = K.variable(0.)

# Initialize Encoder Model
latent_space_dim = 33

# Directory location
dir = 'VAE_TCGA/Model_saved/vae_RawData_NOAnnealing_Ksum_LatentDimension33/'

if not os.path.exists(dir):
    os.mkdir(dir)

# Sampling function as a layer
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
    return random_sample


# VAE loss function
class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight

    def on_epoch_end(self, epoch, logs={}):
        if epoch > klstart:
            new_weight = min(K.get_value(self.weight) + (1. / kl_annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(K.get_value(self.weight)))


def vae_reconstruction_loss(y_true, y_predict):
    experimental_run_tf_function = False
    reconstruction_loss_factor = 15363
    reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_predict))
    return 0.5*reconstruction_loss_factor * reconstruction_loss


def vae_kl_loss(encoder_mu, encoder_log_variance):
    experimental_run_tf_function = False
    kl_loss = -0.5 * K.sum(1 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance))
    return 0.5*kl_loss


def loss(weight=None):
    experimental_run_tf_function = False

    def vae_loss(true, pred):
        reconstruction_loss = mse(K.flatten(true), K.flatten(pred))
        reconstruction_loss *= 15363
        kl_loss = -0.5 * K.sum(1 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance))
        # kl_loss=vae_kl_loss
        if weight is None:
            return K.mean(reconstruction_loss+kl_loss)
        if weight is not None:
            return (0.5 * reconstruction_loss) + (weight * 0.5 * kl_loss)

    return vae_loss


# Plot the scatterplots
def plot_figure(df, tsne_results, tumor_name, filename=None):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    data=df, s=20).set(title="Scatter Plot")

    means = np.vstack([tsne_results[tumor_name == i].mean(axis=0) for i in (df.y.tolist())])

    sns.scatterplot(means[:, 0], means[:, 1], hue=df.y.tolist(), s=20, ec='black', legend=False)

    for j, i in enumerate(np.unique(df.y.tolist())):
        plt.annotate(i, df.loc[df['y'] == i, ['comp-1', 'comp-2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='top',
                     size=12, weight='bold')
    plt.savefig(filename)


# Load Data, TCGA gene expression data (9059,15363) (tumor sample, gene name)
# This data is tpm (transcripts per million)
data_tcga_exp, _, _, _ = load_data("tcga_exp_data.txt")

# Load Tumor labels
tumor_name_samples = load_data("tcga_sample_tumor.csv")
tumor_name_samples = np.array(tumor_name_samples[3])

# Normalize per feature
# X_tcga = preprocessing.normalize(data_tcga_exp, axis=0)
X_tcga = data_tcga_exp

# Labeling
label_encoder = LabelEncoder()
tumor_name_samples = np.array(tumor_name_samples)
label_encoder.fit(tumor_name_samples)
# print(label_encoder.classes_)
tumor_name_samples_code = label_encoder.transform(tumor_name_samples)
# unique, counts = np.unique(tumor_name_samples_code, return_counts=True)

# Split data into training, testing and validation
X_train, X_test, y_train, y_test = train_test_split(X_tcga, tumor_name_samples, test_size=0.10, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=51)


x = Input(shape=(15363,), name="encoder_input")

encoder_layer1 = Dense(512, name='encoder_layer_1')(x)
encoder_norm_layer1 = BatchNormalization(name="encoder_norm_1")(encoder_layer1)
encoder_activ_layer1 = LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)

# encoder_layer2 = Dense(512, name='encoder_layer_2')(encoder_activ_layer1)
# encoder_norm_layer2 = BatchNormalization(name="encoder_norm_2")(encoder_layer2)
# encoder_activ_layer2 = LeakyReLU(name="encoder_leakyrelu_2")(encoder_norm_layer2)

encoder_mu = Dense(units=latent_space_dim, name="encoder_mu")(encoder_activ_layer1)
encoder_log_variance = Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_activ_layer1)

z = Lambda(sampling, name='z_sample')([encoder_mu, encoder_log_variance])

encoder_model = Model(x, [encoder_mu, encoder_log_variance, z], name="encoder_model")

# Initialize Decoder Model
decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")

# decoder_dense_layer1 = Dense(units=512, name="decoder_dense_1")(decoder_input)
# decoder_norm_layer1 = BatchNormalization(name="decoder_norm_1")(decoder_dense_layer1)
# decoder_activ_layer1 = LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

decoder_dense_layer2 = Dense(units=512, name="decoder_dense_2")(decoder_input)
decoder_norm_layer2 = BatchNormalization(name="decoder_norm_2")(decoder_dense_layer2)
decoder_activ_layer2 = LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

decoder_dense_layer3 = Dense(units=15363, name="decoder_dense_3")(decoder_activ_layer2)
decoder_norm_layer3 = BatchNormalization(name="decoder_norm_3")(decoder_dense_layer3)
decoder_output = LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

# decoder_dense_layer2 = Dense(units=1024, name="decoder_dense_2")(decoder_activ_layer1)
# decoder_norm_layer2 = BatchNormalization(name="decoder_norm_2")(decoder_dense_layer2)
# decoder_activ_layer2 = LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

# decoder_dense_layer3 = Dense(units=15363, name="decoder_dense_3")(decoder_activ_layer2)
# decoder_norm_layer3 = BatchNormalization(name="decoder_norm_3")(decoder_dense_layer3)
# decoder_output = LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

decoder_model = Model(decoder_input, decoder_output, name="decoder_model")

# Define VAE
# vae_input = Input(shape=(15363,), name="VAE_input")
vae_encoder_output = encoder_model(x)
vae_decoder_output = decoder_model(encoder_model(x)[2])
vae = Model(x, vae_decoder_output, name="VAE")

# For training enable this
if 1 == 1:
    # Training
    opt = tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-3)
    # vae.compile(optimizer=opt, loss=vae_reconstruction_loss)
    # history1 = vae.fit(X_train, X_train, epochs=35, batch_size=64, shuffle=False)
    vae.compile(optimizer=opt, loss=loss(weight), metrics=[vae_reconstruction_loss, vae_kl_loss])
    history2 = vae.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_val, X_val), batch_size=64, shuffle=False)
    # history2 = vae.fit(X_train, X_train, epochs=n_epochs, batch_size=64, validation_data=(X_val, X_val),
    #                    callbacks=[(AnnealingCallback(weight))], shuffle=False)

    # save model
    encoder_model.save([dir + 'a_encoder.h5'][0])
    decoder_model.save([dir + 'a_decoder.h5'][0])
    vae.save([dir + 'a_vae.h5'][0])

    # Plot the training and validation loss
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title("Model Loss (MSE Reconstruction Loss)")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Val'])
    plt.savefig([dir + 'Figure_ModelLoss_MSE.png'][0])
    plt.show()

    # Plot RL and KL loss during training
    plt.plot(history2.history['vae_reconstruction_loss'])
    plt.title('Training Reconstruction Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig([dir + 'Figure_Training_RL.png'][0])
    plt.show()

    plt.plot(history2.history['vae_kl_loss'])
    plt.title('Training KL Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig([dir + 'Figure_Training_KL.png'][0])
    plt.show()


# To load an existing model
if 1 == 0:
    encoder_model.load_weights([dir + 'a_encoder.h5'][0])
    decoder_model.load_weights([dir + 'a_decoder.h5'][0])
    vae.load_weights([dir + 'a_vae.h5'][0])

# Latent space visualization
X_test_mu, X_test_std, X_test_sample = encoder_model.predict(X_test)
if latent_space_dim > 2:
    X_test_mu = TSNE(n_components=2, perplexity=40).fit_transform(X_test_mu)
df = pd.DataFrame()
df["y"] = y_test
df["comp-1"] = X_test_mu[:, 0]
df["comp-2"] = X_test_mu[:, 1]
plot_figure(df, X_test_mu, y_test,
            filename=[dir + 'Figure_TestData_LatentSpaceVisualization.png'][0])

X_tcga_mu, X_tcga_std, X_tcga_dsample = encoder_model.predict(X_tcga)
if latent_space_dim > 2:
    X_tcga_mu = TSNE(n_components=2, perplexity=40).fit_transform(X_tcga_mu)
df = pd.DataFrame()
df["y"] = tumor_name_samples
df["comp-1"] = X_tcga_mu[:, 0]
df["comp-2"] = X_tcga_mu[:, 1]
plot_figure(df, X_tcga_mu, tumor_name_samples,
            filename=[dir + 'Figure_TCGAData_LatentSpaceVisualization.png'][0])

# Decode the latent space representation of test dataset
X_test_decoded = decoder_model.predict(X_test_sample)
X_tcga_decoded = decoder_model.predict(X_tcga_dsample)

# TSNE projections of test dataset and decoded test dataset
tsne = TSNE(n_components=2, verbose=1, perplexity=30)
tsne_test_results = tsne.fit_transform(X_test)
df = pd.DataFrame()
df["y"] = y_test
df["comp-1"] = tsne_test_results[:, 0]
df["comp-2"] = tsne_test_results[:, 1]
# plt.figure(figsize=(10, 10))
plot_figure(df, tsne_test_results, y_test,
            filename=[dir + 'Figure_RawTestData_TSNEProjection.png'][0])

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_results_decoded = tsne.fit_transform(X_test_decoded)
df = pd.DataFrame()
df["y"] = y_test
df["comp-1"] = tsne_results_decoded[:, 0]
df["comp-2"] = tsne_results_decoded[:, 1]
plot_figure(df, tsne_results_decoded, y_test,
            filename=[dir + 'Figure_VAEDecoderTestData_TSNEProjection.png'][0])

k = 1
