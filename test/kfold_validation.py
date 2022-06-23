import imp
import sys

import networkx as nx
import numpy as np
import pandas as pd
from algorithms.diamond import DIAMOnD
from algorithms.pdiamond_log import pDIAMOnD_log
from algorithms.heat_diffusion import run_heat_diffusion
from utils.network_utils import *
from utils.metrics_utils import *
from utils.data_utils import *


# cross validation
def k_fold_cross_validation(network, algorithm, disease_name, seed_genes, K=5, database_name=None, hyperparams=None):
    '''
    K-Fold Cross Validation.

    Args
    ----
        - network
        - algorithm
        - disease_name
        - seed_genes
        - K
        - database_name
        - hyperparams

    Return
    ------
        print the scores DataFrame in a file
    '''

    # get all genes in the network
    all_genes = list(network.nodes)

    # split  list in K equal parts
    splitted_disease_genes = split_list(seed_genes, K)

    # how many genes predict
    num_genes_to_predict = 200

    # K-fold cross validation:
    # Compute the score for each algorithm using the top 25, top 50, top 100 and top 200 predicted genes
    print(f"{K}-Fold Cross Validation of {algorithm.upper()} on {disease_name.upper()}")

    # init scores array
    scores = np.zeros((K,4,4))

    for k in range(K):
        # init predicted genes and scores
        predicted_genes = []

        # split list and get the k-th
        print("===============================")
        print(f"iteration {k+1}/{K}")

        disease_genes = splitted_disease_genes.copy()
        test_genes = disease_genes.pop(k)
        training_genes = [gene for sublist in disease_genes for gene in sublist] # flatten the list of lists


        # if the algorithm doesn't return a ranking set this flag False
        ranking_flag = True

        # Run algorithm
        if algorithm == "diamond":
            predicted_genes_outfile = f"predicted_genes/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__kfold_{k+1}_{K}.txt"
            csv_outfile = f"results/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{K}_fold.csv"

            added_nodes = DIAMOnD(network, training_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile)
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "pdiamond_log":
            n_iters = hyperparams["pdiamond_n_iters"]
            temp = hyperparams["pdiamond_temp"]
            top_p = hyperparams["pdiamond_top_p"]
            top_k = hyperparams["pdiamond_top_k"]

            predicted_genes_outfile = f"predicted_genes/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{n_iters}_iters__temp_{temp}__top_p_{top_p}__top_k_{top_k}__kfold_{k+1}_{K}.txt"
            csv_outfile = f"results/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{n_iters}_iters__temp_{temp}__top_p_{top_p}__top_k_{top_k}__{K}_fold.csv"

            added_nodes = pDIAMOnD_log(network, training_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile, max_num_iterations=n_iters, temperature=temp, top_p=top_p, top_k=top_k)
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "heat_diffusion":
            diffusion_time = hyperparams["heat_diffusion_time"]

            predicted_genes_outfile = f"predicted_genes/{database_name}/kfold/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-diff_time_{diffusion_time}-kfold_{k+1}_{K}.txt"
            csv_outfile = f"results/{database_name}/kfold/{algorithm}/{algorithm}-{string_to_filename(disease_name)}-diff_time_{diffusion_time}-{K}_fold.csv"

            predicted_genes = run_heat_diffusion(network, training_genes, n_positions=num_genes_to_predict, diffusion_time=diffusion_time)

        else:
            print("  ERROR: No valid algorithm.     ")
            print("  Choose one of the following:   ")
            print("    - diamond                    ")
            print("    - pdiamond_log               ")
            print("    - heat_diffusion             ")
            sys.exit(1)

        # compute the scores over the predicted genes
        scores[k] = np.array((compute_metrics(all_genes, test_genes, predicted_genes[:25]),
                                compute_metrics(all_genes, test_genes, predicted_genes[:50]),
                                compute_metrics(all_genes, test_genes, predicted_genes[:100]),
                                compute_metrics(all_genes, test_genes, predicted_genes[:200]))).transpose()

        # print iteration results
        print(scores[k])
        print("===============================")

    # compute average and std deviation of the metrics
    scores_avg = np.average(scores, axis=0)
    scores_std = np.std(scores, axis=0)

    print(f"scores_avg:\n{scores_avg}")
    print(f"score_std:\n{scores_std}")

    # create the final score dataframe where each value is (avg, std)
    n_rows, n_cols = scores_avg.shape
    final_scores = np.zeros((n_rows, n_cols), dtype=tuple)

    for i in range(n_rows):
        for j in range(n_cols):
            final_scores[i][j] = (scores_avg[i][j], scores_std[i][j])

    # create a DataFrame with the results
    metrics = ["precision", "recall", "f1", "ndcg"]
    sizes = ["Top 25","Top 50", "Top 100", "Top 200"]
    result_df = pd.DataFrame(final_scores, metrics, sizes)

    result_df.to_csv(csv_outfile)
