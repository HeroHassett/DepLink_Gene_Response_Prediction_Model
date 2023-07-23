import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
from import_the_data import load_data, load_geneset

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model, Sequential
from keras.layers import Dense, LeakyReLU

import datetime
from datetime import datetime
from packaging import version

from hyperas import optim

from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform

"""
    To run this code simply put all the files that the code "loads" into the same folder as this code.
    Next you can simply change the hyperparameters you want optimized
    Finally if you would like to add a different model you would do so in the "create_model" function
    Then call the create_model function in the optim.minimize function and that function will optimize your model
"""

# Input Variables
input_dim = 4682
latent_dim = 64

# Load the TensorBoard notebook extension
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

def data():
    # Load gene expression
    data_exp, _, sample_names_exp, gene_names_exp = load_data("/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/gene_DepMap_21Q4/ccle_exp_data.txt")
    data_mut, _, sample_names_mut, gene_names_mut = load_data("/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/gene_DepMap_21Q4/ccle_mut_data.txt")
    data_drug_lf = np.array(pd.read_csv("/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/prism_matrix.csv"))

    cell_ids = np.array(range(data_exp.shape[0]))

    # Create training, test, and validation data sets using 80% and 20% split
    train_idx, test_idx, _, _ = train_test_split(cell_ids, cell_ids, test_size=0.20, random_state=101)
    train_idx, val_idx, _, _ = train_test_split(train_idx, train_idx, test_size=0.05, random_state=101)

    # Make a label data set for all the gene expressions and gene mutations
    data_exp_train = data_exp[train_idx, :]
    data_exp_test = data_exp[test_idx, :]
    data_exp_val = data_exp[val_idx, :]

    data_mut_train = data_mut[train_idx, :]
    data_mut_test = data_mut[test_idx, :]
    data_mut_val = data_mut[val_idx, :]

    drug_lf_train = data_drug_lf[train_idx, :]
    drug_lf_test = data_drug_lf[test_idx, :]
    drug_lf_val = data_drug_lf[val_idx, :]

    return data_exp_train, data_mut_train, drug_lf_train, data_exp_test, data_mut_test, drug_lf_test, data_exp_val, data_mut_val, drug_lf_val

# Make the neural network
def create_model(data_exp_train, data_mut_train, drug_lf_train, data_exp_val, data_mut_val, drug_lf_val):
    # Establish input variables
    activation_func = 'leaky_relu'
    activation_func2 = 'linear'
    init = 'he_uniform'
    batch_size = {{choice([32, 64, 128])}}
    num_epoch = {{choice([50, 100, 150, 200])}}

    # Load weights of the TCGA pretrained autoencoder
    tcga_geneExp_model = pickle.load(open(
        "/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/gene_DepMap_21Q4"
        "/tcga_pretrained_autoencoder_geneExp.pickle",
        "rb"))

    # Create the feature extractor neural network for the CCLE data set
    ccle_feature_extractorE = Sequential()
    ccle_feature_extractorE.add(
        Dense(1024, input_shape=(15363,), weights=tcga_geneExp_model[0], name='feature_extractorE_layer1'))
    ccle_feature_extractorE.add(LeakyReLU(weights=tcga_geneExp_model[1], name='feature_extractorE_layer2'))
    ccle_feature_extractorE.add(Dense(256, weights=tcga_geneExp_model[2], name='feature_extractorE_layer3'))
    ccle_feature_extractorE.add(LeakyReLU(weights=tcga_geneExp_model[3], name='feature_extractorE_layer4'))
    ccle_feature_extractorE.add(Dense(64, weights=tcga_geneExp_model[4], name='feature_extractorE_layer5'))
    ccle_feature_extractorE.add(LeakyReLU(weights=tcga_geneExp_model[5], name='feature_extractorE_layer6'))
    ccle_feature_extractorE.add(Dense(4686, name='outputE_layer'))

    # Gene Mutation Feature Extractor (Input Layer + 2 Dense Layers)
    tcga_mutExp_model = pickle.load(open(
        "/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/gene_DepMap_21Q4"
        "/tcga_pretrained_autoencoder_geneMutation.pickle",
        "rb"))
    ccle_feature_extractorM = Sequential()
    ccle_feature_extractorM.add(
        Dense(1024, input_shape=(18281,), weights=tcga_mutExp_model[0], name='feature_extractorM_layer1'))
    ccle_feature_extractorM.add(LeakyReLU(weights=tcga_mutExp_model[1], name='feature_extractorM_layer2'))
    ccle_feature_extractorM.add(Dense(256, weights=tcga_mutExp_model[2], name='feature_extractorM_layer3'))
    ccle_feature_extractorM.add(LeakyReLU(weights=tcga_mutExp_model[3], name='feature_extractorM_layer4'))
    ccle_feature_extractorM.add(Dense(64, weights=tcga_mutExp_model[4], name='feature_extractorM_layer5'))
    ccle_feature_extractorM.add(LeakyReLU(weights=tcga_mutExp_model[5], name='feature_extractorM_layer6'))
    ccle_feature_extractorM.add(Dense(4686, name='outputM_layer'))

    # Merge the two feature extractors
    merged_output = keras.layers.concatenate([ccle_feature_extractorE.output, ccle_feature_extractorM.output])

    # Make single prediction neural network
    model_combined = Sequential()
    model_combined.add(
        Dense({{choice([16384, 8192, 4096])}}, activation=activation_func, kernel_initializer=init, name="Comb_dense1"))
    model_combined.add(
        Dense({{choice([2048, 1024])}}, activation=activation_func, kernel_initializer=init, name="Comb_dense2"))
    model_combined.add(Dense(104, activation=activation_func2, kernel_initializer=init, name="Comb_dense3"))

    model_final = Model([ccle_feature_extractorE.input, ccle_feature_extractorM.input], model_combined(merged_output))

    # Compile and fit the model on the training and validation data set
    model_final.compile(loss='mse', optimizer='adam')
    history = model_final.fit([data_exp_train, data_mut_train], drug_lf_train, epochs=num_epoch,
                              validation_data=([data_exp_val, data_mut_val], drug_lf_val),
                              batch_size=batch_size,
                              shuffle=True, verbose=2)
    validation_loss = np.amin(history.history['val_loss'])

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model_final}

# Hyperas optimization of the model
if __name__ == '__main__':
    batch_size = 32
    # Optimize the model with the goal of minimizing the validation loss
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=40,
                                          trials=Trials(), eval_space=True, keep_temp=True, verbose=True)
    data_exp_train, data_mut_train, drug_lf_train, data_exp_test, data_mut_test, drug_lf_test, data_exp_val, data_mut_val, drug_lf_val = data()
    print("Evaluation of best performing model:")
    # Evaluate the best model
    print(best_model.evaluate([data_exp_test, data_mut_test], drug_lf_test, verbose=0, batch_size=batch_size))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)