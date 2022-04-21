library(tidyr)
library(rcdk)
library(rJava)
library(readxl)
library(readr)
library("stringr")

# Generate codes for CCLE data set 265 drugs
load("Data/drug_PRISM_2019/primary-screen-replicate-collapsed-logfold-change_treatment-info_Tapsya.rdata")
prism_smiles <- data.frame('smiles'=prism_info$smiles)
prism_smiles <-drop_na(prism_smiles)
prism_smiles <- str_split(prism_smiles$smiles, ',')
prism_smiles <- lapply(prism_smiles, "[[", 1)
prism_smiles <- as.character(prism_smiles)

# Get the fingerprint code
mols <- parse.smiles(prism_smiles[[9]])

# For standard and extended use this one
fps <- lapply(mols, get.fingerprint, type="pubchem") # "maccs"
number_drugs <- length(mols)
matrix_fp <- matrix(0, number_drugs, 881) #166
for (i in 1:number_drugs){
  matrix_fp[i,fps[[i]]@bits] = 1
}

prism_smiles <- data.frame("drug_name"=prism_smiles, "smiles"=prism_smiles,
                          "binary"=matrix_fp)
write.csv(prism_smiles,file="info_smiles2binary_881bits_pubchem.csv")

