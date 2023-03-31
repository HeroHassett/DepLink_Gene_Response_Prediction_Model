# from os import listdir
# from os.path import isfile, join
# from collections import defaultdict
import numpy as np
import pickle

def load_data(filename, log_trans=False, label=False):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    # labels = lines[1].replace('\n', '').split('\t')[1:]
    if label:
        data_labels = lines[1].replace('\n', '').split('\t')[1:]
        dx = 2
    else:
        dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        # print gene
        gene_names.append(gene)
        # print values[1:]
        data.append(values[1:])
        # print len(values)
    data = np.array(data, dtype='float32')
    # print all_data.shape
    if log_trans:
        data = np.log2(data + 1)
    data = np.transpose(data)

    # print all_label.shape
    # print labeled_label.shape
    # print count
    # print labeled_data
    # print labeled_data.shape
    # print unlabeled_data.shape
    # print all_data
    # print all_data.shape
    # print gene_names
    return data, data_labels, sample_names, gene_names

# data, data_labels, sample_names, gene_names = load_data("Merge_FCscranERCCnorm_TPM_filtered.txt")
# geneset_file = "BioCarta.gmt"

def load_geneset(gene_names, geneset_file):
    geneset_names = []
    geneset_lines = open(geneset_file).readlines()
    # genes_idx = defaultdict(lambda:[])
    geneset_mat = np.zeros((len(geneset_lines), len(gene_names)),dtype="float32")
    for ix, line in enumerate(geneset_lines):
        info = line.replace('\n','').replace('\r','').split('\t')
        geneset_names.append(info[0])
        for sp in info[2:]:
            if sp in gene_names:
                geneset_mat[ix, gene_names.index(sp)] = 1
    return geneset_mat, geneset_names
