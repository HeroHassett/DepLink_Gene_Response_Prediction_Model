# Import packages
library(rcdk)
library(rJava)
library(tidyr)
library(readr)
library("stringr")

# Load PRISM & CCLE data
load('Data/drug_PRISM_2019/primary-screen-replicate-collapsed-logfold-change_treatment-info_Tapsya.rdata')
gene_exp <- readRDS('Data/gene_DepMap_21Q4/21Q3/CCLE_exp_organized.RDS')
ccle_data <- read_delim("/Users/alexk/Documents/GitHub/DepLink_Gene_Response_Prediction_Model/Data/gene_DepMap_21Q4/ccle_exp_data.txt")
ccle_data <- ccle_data[,!(names(ccle_data) %in% "...1")]
ccle_data <- ccle_data[1:104]
ccle_data <- t(ccle_data)

# Switch prism_data column for the prism_info broad_id column
colnames(prism_data) <- prism_info$broad_id

# transpose CCLE/gene_exp matrix
gene_exp <- t(gene_exp)

# Make the rows of prism_data and gene_exp the same
cell_ids <- intersect(rownames(prism_data), rownames(gene_exp))

# Input intersected data values into their respective data frames
prism_data_new <- prism_data[cell_ids, ]
gene_exp_new <- gene_exp[cell_ids, ]

# Remove NA values from prism_data_new
prism_data_new <- drop_na(prism_data_new)

# Remove the gene expressions that had null values for their drug response
gene_exp_data <- merge(gene_exp_new, prism_data_new, by=0)
rownames(gene_exp_data) <- gene_exp_data[,1]
gene_exp_data <- gene_exp_data[,!(names(gene_exp_data) %in% "Row.names")]
gene_exp_data <- gene_exp_data[1:17040]