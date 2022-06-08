import sys

import networkx as nx
import numpy as np
import pandas as pd
from algorithms.diamond import DIAMOnD
from algorithms.pdiamond import pDIAMOnD
from algorithms.pdiamond_log import pDIAMOnD_log
# from algorithms.pdiamond_entropy import pDIAMOnD_entropy
from algorithms.heat_diffusion import run_heat_diffusion
from utils.network_utils import *
from utils.metrics_utils import *
from utils.data_utils import *


def extended_validation(network, algorithm, disease_name, seed_genes, test_genes, database_name=None, hyperparams=None):
    '''
    Perform an extended validation using all the seed genes of the disease
    and test the predicted genes with all the genes in
    all_disease_gene_associations.tsv
    '''

    # get all genes in the network
    all_genes = list(network.nodes)

    # how many genes predict
    num_genes_to_predict = 200

    # Extended Validation:
    # compute the score for each algorithm using the top 25, top 50, top 100 and top 200 predicted genes
    print(f"Extended validation of {algorithm.upper()} on {disease_name.upper()}...", end="\n\n")

    # if the algorithm doesn't return a ranking set this flag False
    ranking_flag = True

    if algorithm == "diamond":
        predicted_genes_outfile = f"predicted_genes/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-extended.txt"
        csv_outfile = f"results/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-extended.csv"

        added_nodes = DIAMOnD(network, seed_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond":
        n_iters = hyperparams["pdiamond_n_iters"]

        predicted_genes_outfile = f"predicted_genes/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-extended.txt"
        csv_outfile = f"results/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-extended.csv"

        added_nodes = pDIAMOnD(network, seed_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile, max_num_iterations=n_iters)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond_log":
        n_iters = hyperparams["pdiamond_n_iters"]
        temp = hyperparams["pdiamond_temp"]
        top_p = hyperparams["pdiamond_top_p"]
        top_k = hyperparams["pdiamond_top_k"]

        predicted_genes_outfile = f"predicted_genes/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-temp_{temp}-top_p_{top_p}-top_k_{top_k}-extended.txt"
        csv_outfile = f"results/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-temp_{temp}-top_p_{top_p}-top_k_{top_k}-extended.csv"

        added_nodes = pDIAMOnD_log(network, seed_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile, max_num_iterations=n_iters, temperature=temp, top_p=top_p, top_k=top_k)
        predicted_genes = [item[0] for item in added_nodes]

    # elif algorithm == "pdiamond_entropy":
    #     n_iters = hyperparams["pdiamond_n_iters"]
    #     temp = hyperparams["pdiamond_temp"]
    #     top_p = hyperparams["pdiamond_top_p"]
    #     top_k = hyperparams["pdiamond_top_k"]

    #     predicted_genes_outfile = f"predicted_genes/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-temp_{temp}-top_p_{top_p}-top_k{top_k}-extended.txt"
    #     csv_outfile = f"results/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-{n_iters}_iters-temp_{temp}-top_p_{top_p}-top_k_{top_k}-extended.csv"

    #     added_nodes = pDIAMOnD_entropy(network, seed_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile, max_num_iterations=n_iters, temperature=temp, top_p=top_p, top_k=top_k)
    #     predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "heat_diffusion":
        diffusion_time = hyperparams["heat_diffusion_time"]

        predicted_genes_outfile = f"predicted_genes/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-diff_time_{diffusion_time}-extended.txt"
        csv_outfile = f"results/{database_name}/extended/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-diff_time_{diffusion_time}-extended.csv"

        predicted_genes = run_heat_diffusion(network, seed_genes, n_positions=num_genes_to_predict, diffusion_time=diffusion_time)

    else:
        print("  ERROR: No valid algorithm.    ")
        print("  Choose one of the following:  ")
        print("    - diamond                   ")
        print("    - pdiamond                  ")
        print("    - pdiamond_log              ")
        # print("    - pdiamond_entropy          ")
        print("    - heat_diffusion            ")
        sys.exit(1)

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

    result_df.to_csv(csv_outfile)
