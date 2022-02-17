import sys

import networkx as nx
import numpy as np
import pandas as pd
from algorithms.diamond import run_diamond
from algorithms.prob_diamond import run_prob_diamond
from utils.network_utils import *
from utils.metrics_utils import *
from utils.data_utils import *


def extended_validation(network, seed_genes, test_genes, algorithm_name, disease_name, num_iters_prob_diamond=1):
    '''
    Perform an extended validation using all the seed genes of the disease
    and test the predicted genes with all the genes in
    all_disease_gene_associations.tsv
    '''

    # list of algorithms on which run the extended validation
    if algorithm_name == "all":
        algorithm_list = ["diamond",
                          "prob_diamond"]
    else:
        algorithm_list = [algorithm_name]

    # get all genes in the network
    all_genes = list(network.nodes)

    # get number of seed genes
    num_disease_genes = len(seed_genes)

    # write network connections in a file
    interactome = "data/HHI_LCC.txt"
    write_edges_in_file(network, interactome)

    # write seed genes in a file
    seed_genes_file = "tmp/seed_genes.txt"
    with open(seed_genes_file, 'w') as sf:
        for gene in seed_genes:
            sf.write(gene + "\n")

    # define how many genes to predict
    # (predicting at least 200 genes)
    if num_disease_genes < 200:
        num_genes_to_predict = 200
    else:
        num_genes_to_predict = num_disease_genes

    # compute the score for each algorithm using the top 50, top 100, top 200 and top N predicted genes
    # where N = num genes in curated GDAs
    for algorithm in algorithm_list:
        print(f"Extended validation of {algorithm.upper()} on {disease_name.upper()}...", end="\n\n")

        # output file for diamond-like algorithms
        output_file = f"tmp/{algorithm}_output.txt"

        predicted_genes = []

        if algorithm == "diamond":
            input_list = ["DIAMOnD.py", interactome, seed_genes_file, num_genes_to_predict, 1, output_file]
            added_nodes = run_diamond(input_list)   # take only first elements of each sublist
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "prob_diamond":
            input_list = ["prob_diamond.py", interactome, seed_genes_file, num_genes_to_predict, 1, output_file, num_iters_prob_diamond]
            added_nodes = run_prob_diamond(input_list)
            predicted_genes = [item[0] for item in added_nodes]

        else:
            print("  ERROR: No valid algorithm.  ")
            print("  Choose one of the following:")
            print("    - diamond                 ")
            print("    - prob_diamond            ")
            sys.exit(0)

        # compute the scores over the predicted genes
        scores = np.array((compute_metrics(all_genes, seed_genes, predicted_genes[:50], test_genes),
                           compute_metrics(all_genes, seed_genes, predicted_genes[:100], test_genes),
                           compute_metrics(all_genes, seed_genes, predicted_genes[:200], test_genes),
                           compute_metrics(all_genes, seed_genes, predicted_genes[:num_disease_genes], test_genes))).transpose()

        # create a DataFrame with the results
        metrics = ["precision", "recall", "f1", "ndcg"]
        sizes = ["Top 50","Top 100", "Top 200", "Top N"]
        result_df = pd.DataFrame(scores, metrics, sizes)

        # save it as csv file
        csv_file = f"results/extended/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_extended.csv"
        result_df.to_csv(csv_file)
