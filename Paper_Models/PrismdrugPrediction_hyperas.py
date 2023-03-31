# Prism Drug Prediction 3 October 2022
# Common cell lines between ccle and prism data: 298
# Prism data is imputed using MICE
# Note: Hyperas you cannot keep comments in the model function as it does not read it.

import keras
import pickle
import csv
import re
import numpy as np
import tensorflow as tf
from hyperas import optim
import matplotlib.pyplot as plt
from keras.layers import merge
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from keras.models import Model, Sequential, load_model
from keras.callbacks import Callback, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU, ReLU

from import_the_data import load_data, load_geneset

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding, Permute, \
    Dropout, BatchNormalization, add, concatenate, Concatenate

def data():
    # Load gene expression
    data_exp, _, sample_names_exp, gene_names_exp = load_data("ccle_exp_data.txt")
    data_mut, _, sample_names_mut, gene_names_mut = load_data("ccle_mut_data.txt")
    data_drug_lf, _, sample_names_pr, names_drug = load_data("prism_data_pr.txt")

    cell_ids = np.array(range(data_exp.shape[0]))

    train_idx, test_idx, _, _ = train_test_split(cell_ids, cell_ids, test_size=0.20, random_state=101)
    train_idx,  val_idx, _, _ = train_test_split(train_idx, train_idx, test_size=0.05, random_state=101)

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


# save model parameters to pickle
def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)


def create_model(data_exp_train, data_mut_train, drug_lf_train, data_exp_val, data_mut_val, drug_lf_val):
    # Set hyperparameters
    activation_func = 'leaky_relu'
    activation_func2 = 'linear'
    init = 'he_uniform'
    batch_size = {{choice([32, 64, 128])}}
    num_epoch = {{choice([50, 100, 150, 200])}}

    # load the pretrained models
    tcgaExp_modelweights = pickle.load(open("tcga_exp_data_1024_256_64_64_100.pickle", "rb"))
    tcgaMut_modelweights = pickle.load(open("tcga_mut_data_1024_256_64_64_100.pickle", "rb"))

    # Autoencoder model for gene expression
    autoencoderExp_model = Sequential()
    autoencoderExp_model.add(Dense(1024, input_shape=(15363,), weights=tcgaExp_modelweights[0]))
    autoencoderExp_model.add(ReLU())
    autoencoderExp_model.add(Dense(256, weights=tcgaExp_modelweights[1]))
    autoencoderExp_model.add(ReLU())
    autoencoderExp_model.add(Dense(64, weights=tcgaExp_modelweights[2]))
    autoencoderExp_model.add(ReLU())

    # Autoencoder model for gene mutation
    autoencoderMut_model = Sequential()
    autoencoderMut_model.add(Dense(1024, input_shape=(18281,), weights=tcgaMut_modelweights[0]))
    autoencoderMut_model.add(ReLU())
    autoencoderMut_model.add(Dense(256, weights=tcgaMut_modelweights[1]))
    autoencoderMut_model.add(ReLU())
    autoencoderMut_model.add(Dense(64, weights=tcgaMut_modelweights[2]))
    autoencoderMut_model.add(ReLU())

    merged_output = concatenate([autoencoderExp_model.output, autoencoderMut_model.output])

    model_combined = Sequential()
    model_combined.add(Dense({{choice([16384, 8192, 4096])}}, activation=activation_func, kernel_initializer=init, name="Comb_dense1"))
    model_combined.add(Dense({{choice([2048, 1024])}}, activation=activation_func, kernel_initializer=init, name="Comb_dense2"))
    model_combined.add(Dense(4684, activation=activation_func2, kernel_initializer=init, name="Comb_dense3"))

    model_final = Model([autoencoderExp_model.input, autoencoderMut_model.input], model_combined(merged_output))

    #model_final.summary()

    model_final.compile(loss='mse', optimizer='adam')
    history = model_final.fit([data_exp_train, data_mut_train], drug_lf_train, epochs=num_epoch,
                              validation_data=([data_exp_val, data_mut_val], drug_lf_val), batch_size=batch_size, shuffle=True, verbose=2)
    validation_loss = np.amin(history.history['val_loss'])

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model_final}


if __name__ == '__main__':
    batch_size = 32
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=40, trials=Trials(), eval_space=True, keep_temp=True, verbose=True)
    data_exp_train, data_mut_train, drug_lf_train, data_exp_test, data_mut_test, drug_lf_test, data_exp_val, data_mut_val, drug_lf_val = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate([data_exp_test, data_mut_test], drug_lf_test, verbose=0, batch_size=batch_size))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    save_weight_to_pickle(best_model, 'prismDrugResponse_hyperas_bestModel.pkl')