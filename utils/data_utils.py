
import enum
from fnmatch import translate
from ossaudiodev import control_names
import sys
import csv
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from pytest import skip
from sklearn.utils import shuffle

#################
#   UTILITIES   #
#################

def translate_gene_names(genes=None, database=None):
    if database == "biogrid":
        return genes

    if database == "stringdb":
        # Read the alisases from StringDB
        stringdb_aliases = pd.read_csv("data/9606.protein.aliases.v11.5.txt", sep="\t", header=0)

        # Select only the rows that have
        # 1. the alias name in our genes list
        # 2. the source equals to Ensembl_HGNC
        stringdb_aliases = stringdb_aliases.loc[(stringdb_aliases["alias"].isin(genes)) &
                                                (stringdb_aliases["source"] == "Ensembl_HGNC")]
        translated_genes = stringdb_aliases["string_protein_id"].to_list()

        # Remove duplicates
        translated_genes = list(set(translated_genes))

        return translated_genes

def get_num_genes_per_disease(disease_list):
    '''
    Given a list of diseases
    Return a dictionary with (key: disease name, value: num genes)
    '''
    num_genes_per_disaese = {}
    for disease in disease_list:
        num_genes = len(get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv",
                                                   disease))
        num_genes_per_disaese[disease] = num_genes

    return num_genes_per_disaese

def get_disease_genes_from_gda(filename, disease, training_mode=True, translate_in="biogrid"):
    '''
    Find all genes associated to a given disease in the GDA
    and return them as a list.
    '''

    df = pd.read_csv(filename, sep='\t', header=0)

    # select only rows associated to the disease
    disease_df = df.loc[df['diseaseName'] == disease]

    # get all the genes associated to the disease
    disease_genes = disease_df['geneSymbol'].to_list()

    # translate disease gene names wrt the database we are using
    disease_genes = translate_gene_names(genes=disease_genes, database=translate_in)

    # shuffling
    if training_mode == True:   # fixed seed for results reliability
        shuffled_genes = disease_genes.copy()
        random.seed(42)
        random.shuffle(shuffled_genes)
    else:
        shuffled_genes = disease_genes.copy()
        random.shuffle(shuffled_genes)   # modify the original list

    return shuffled_genes

def get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=False):
    """
    Given a file with the seeds and the name of a disease,
    return the disease genes associated with that diseses.
    """
    # Load the seeds file
    df = pd.read_csv(seeds_file, sep="\t", header=0)
    
    # Get the seed genes for the given disease
    seeds_string = df.loc[df["Diseases"]==disease]["Genes"].item()
    seeds = seeds_string.split("/")
    seeds = [int(seed) for seed in seeds]

    # shuffling
    if fix_random == True:   # fixed seed for results reliability
        shuffled_seeds = seeds.copy()
        random.seed(42)
        random.shuffle(shuffled_seeds)
    else:
        shuffled_seeds = seeds.copy()
        random.shuffle(shuffled_seeds)   # modify the original list

    return shuffled_seeds

def split_list(list, n):
    ''' Split a list in n equals parts '''
    splitted_list = []
    n_split = np.array_split(list, n)
    for array in n_split:
        splitted_list.append(array.tolist())

    return splitted_list

def string_to_filename(s):
    ''' Convert a string in a format useful for file names '''
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    s = s.replace(",", "")
    s = s.replace("_-_", "_")
    s = s.lower()
    return s

def fix_metric_name(s):
    '''
    For a clarity of plot title, convert all "avg_score" in "score"
    '''

    if 'avg_' in s:
        return s.split('avg_')[1]
    else:
        return s


def split_result_std(df):
    '''
    Given a data frame with values in the form "avg +/- std".
    Split the averages and the stds, convert to floarnumpy array and return them.
    '''

    values = df.values
    averages = np.zeros(values.shape)
    stds = np.zeros(values.shape)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            averages[i][j], stds[i][j] = [float(values[i][j].split(" +/- ")[0]),
                                          float(values[i][j].split(" +/- ")[1])]

    return np.array(averages), np.array(stds)

def get_score_from_dataframe(df, metric_name, keys=None, multi_index=False):
    '''
    Read a DataFrame with the scores of an algorithm
    and return the values associated to the matric_name.
    If the value is a string convert to float.
    '''

    if multi_index == False:
        # get values associated to the given metric in array fomr
        values = df.loc[metric_name].values

        # init final score of the metric
        num_values = len(values)
        score = np.zeros(num_values)

        # check if the values are strings
        for idx, value in enumerate(values):
            # if string, separate value and std deviations
            if type(value) == str:
                value = float(value.split(" +/- ")[0])
                score[idx] = value
            else:
                score[idx] = value

    if multi_index == True:
        prediction_size = keys[0]
        metric = keys[1]

        df = df.xs(prediction_size, axis=1, level=1, drop_level=False)
        values = df.loc[metric].values

        score = values

    return score

def get_diseases_from_file(disease_file):
    '''
    Read diseases in disease file and
    return their names as a list
    '''
    disease_list = []
    with open(disease_file, 'r') as df:
        for line in df:
            disease_name = line.replace("\n","") # remove new line
            disease_list.append(disease_name)

    return disease_list

def create_disease_mesh_dictionary(disease_file):
    '''
    Read the disease names from disease file
    and retrieve for each disease its MeSH Class.
    Then, create a dictionary with
    Key: MeSH class
    Value: Disease1, Disease2, ...

    Also, save the dictionary on a file
    '''

    # retrieve disease names
    disease_list = []
    with open(disease_file, 'r') as df:
        for line in df:
            disease_name = line.replace("\n","") # remove new line
            disease_list.append(disease_name)

    # ======================================= #
    # Retrieve the mesh class of each disease #
    # ======================================= #

    # 1. import the gene disease associations file as DataFrame
    gda_file = "data/curated_gene_disease_associations.tsv"
    gda_df = df = pd.read_csv(gda_file, sep='\t', header=0)

    # 2. filter out from the df all the disease not in my disease list
    #    and take only 'diseaseName' and 'diseaseClass'
    filtered_gda_df = gda_df[gda_df['diseaseName'].isin(disease_list)]
    filtered_gda_df = filtered_gda_df[['diseaseName', 'diseaseClass']]

    # 3. remove duplicate rows (same diseases)
    filtered_gda_df = filtered_gda_df.drop_duplicates()

    # 4. get disease classes
    disease_classes = filtered_gda_df['diseaseClass'].values

    # 5. some classes are combined (e.g. C06;C23) --> split them
    mesh_classes = []
    for item in disease_classes:
        mesh_codes = item.split(";")
        for code in mesh_codes:
            if code != "":
                mesh_classes.append(code)

    # 6. remove duplicate classes
    mesh_classes = list(set(mesh_classes))

    # =================================== #
    #  Create the mesh-disease dictionary #
    # =================================== #

    # 1. init the dictionary with the mesh codes
    mesh_disease_dict = {}
    for mclass in mesh_classes:
        mesh_disease_dict[mclass] = []

    # 2. append disease to the currespondent MeSH code
    mesh_disease_pairs = filtered_gda_df[['diseaseName', 'diseaseClass']].values
    for mclass in mesh_classes:
        for pair in mesh_disease_pairs:
            if mclass in pair[1]:
                mesh_disease_dict[mclass].append(pair[0])


    # =================================== #
    #      Save dictionary on a file      #
    # =================================== #
    with open("results/disease_by_mesh_class.txt", 'w') as f:
        f.write("MeSH Class\t\tDiseases\n\n")
        classes = mesh_disease_dict.keys()
        for c in classes:
            f.write(f"\t{c}\t\t\t{mesh_disease_dict[c]}\n")

    return mesh_disease_dict
