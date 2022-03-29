import sys

import networkx as nx
import numpy as np
import pandas as pd
from algorithms.diamond import run_diamond
from algorithms.prob_diamond import run_prob_diamond
from algorithms.heat_diffusion import run_heat_diffusion
from utils.network_utils import *
from utils.metrics_utils import *
from utils.data_utils import *


def extended_validation(network, seed_genes, test_genes, algorithm, disease_name, diffusion_time=0.005, num_iters_prob_diamond=10):
    '''
    Perform an extended validation using all the seed genes of the disease
    and test the predicted genes with all the genes in
    all_disease_gene_associations.tsv
    '''

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

    # how many genes predict
    num_genes_to_predict = 200

    # Extended Validation:
    # compute the score for each algorithm using the top 25, top 50, top 100 and top 200 predicted genes
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

    elif algorithm == "heat_diffusion":
        predicted_genes = run_heat_diffusion(network, seed_genes, n_positions=num_genes_to_predict, diffusion_time=diffusion_time)

    else:
        print("  ERROR: No valid algorithm.    ")
        print("  Choose one of the following:  ")
        print("    - diamond                   ")
        print("    - prob_diamond              ")
        print("    - heat_diffusion            ")
        sys.exit(0)

    # compute the scores over the predicted genes
    scores = np.array((compute_metrics(all_genes, test_genes, predicted_genes[:25]),
                        compute_metrics(all_genes, test_genes, predicted_genes[:50]),
                        compute_metrics(all_genes, test_genes, predicted_genes[:100]),
                        compute_metrics(all_genes, test_genes, predicted_genes[:200]))).transpose()

    # print results
    print(scores)

    # create a DataFrame with the results
    metrics = ["precision", "recall", "f1", "ndcg"]
    sizes = ["Top 25","Top 50", "Top 100", "Top 200"]
    result_df = pd.DataFrame(scores, metrics, sizes)

    # save it as csv file
    csv_file = f"results/extended/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_extended.csv"

    # save it as csv file
    if algorithm == "diamond":
        csv_file = f"results/extended/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_extended.csv"

    if algorithm == "prob_diamond":
        csv_file = f"results/extended/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_extended_{num_iters_prob_diamond}_iters.csv"

    if algorithm == "heat_diffusion":
        csv_file = f"results/extended/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_extended_diff_time_{diffusion_time}.csv"


    result_df.to_csv(csv_file)
