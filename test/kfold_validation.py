import sys

import numpy as np
import pandas as pd
from algorithms.diamond import DIAMOnD
from algorithms.proconsul import PROCONSUL
from utils.data_utils import *
from utils.metrics_utils import *
from utils.network_utils import *


def k_fold_cross_validation(network, algorithm, disease_name, seed_genes, K=5, database_name=None, hyperparams=None, all_iterations=False):
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

    # Get all genes in the network.
    all_genes = list(network.nodes)

    # Split the seed genes list in K equal parts.
    splitted_disease_genes = split_list(seed_genes, K)

    # How many genes to predict.
    num_genes_to_predict = 200

    # K-fold cross validation:
    # Compute the score for each algorithm using the top 25, top 50, top 100 and top 200 predicted genes.
    print(f"{K}-Fold Cross Validation of {algorithm.upper()} on {disease_name.upper()}")
    print(f"Hyperparameters: {hyperparams}")
    print(f" ")

    # Init scores array.
    scores = np.zeros((K,4,4))
    if all_iterations == True:
        complete_scores = np.zeros((K, 4, num_genes_to_predict))

    for k in range(K):

        # Init the list of predicted genes.
        predicted_genes = []

        # Split list in K equal parts and get the k-th.
        print("===============================")
        print(f"iteration {k+1}/{K}")

        disease_genes = splitted_disease_genes.copy()
        test_genes = disease_genes.pop(k)
        training_genes = [gene for sublist in disease_genes for gene in sublist] # flatten the list of lists


        # If the algorithm doesn't return a ranking set this flag to False.
        ranking_flag = True

        # Run algorithm.
        if algorithm == "diamond":

            predicted_genes_outfile = f"predicted_genes/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__kfold_{k+1}_{K}.txt"
            csv_outfile = f"results/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{K}_fold.csv"

            added_nodes = DIAMOnD(network, training_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile)
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "proconsul":
            
            # Get the hyperparameters of PROCONSUL
            n_rounds    = hyperparams["proconsul_n_rounds"]
            temp        = hyperparams["proconsul_temp"]
            top_p       = hyperparams["proconsul_top_p"]
            top_k       = hyperparams["proconsul_top_k"]

            predicted_genes_outfile = f"predicted_genes/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{n_rounds}_rounds__temp_{temp}__top_p_{top_p}__top_k_{top_k}__kfold_{k+1}_{K}.txt"
            csv_outfile = f"results/{database_name}/kfold/{algorithm}/{algorithm}__{string_to_filename(disease_name)}__{n_rounds}_rounds__temp_{temp}__top_p_{top_p}__top_k_{top_k}__{K}_fold.csv"

            added_nodes = PROCONSUL(network, training_genes, num_genes_to_predict, 1, outfile=predicted_genes_outfile, n_rounds=n_rounds, temperature=temp, top_p=top_p, top_k=top_k)
            predicted_genes = [item[0] for item in added_nodes]


        else:
            print("  ERROR: No valid algorithm.     ")
            print("  Choose one of the following:   ")
            print("    - diamond                    ")
            print("    - proconsul                  ")
            sys.exit(1)
        
        # Compute the scores over the predicted genes.
        scores[k] = np.array((compute_metrics(all_genes, test_genes, predicted_genes[:25]),
                              compute_metrics(all_genes, test_genes, predicted_genes[:50]),
                              compute_metrics(all_genes, test_genes, predicted_genes[:100]),
                              compute_metrics(all_genes, test_genes, predicted_genes[:200]))).transpose()

        if all_iterations == True:
            complete_scores[k] = np.array([compute_metrics(all_genes, test_genes, predicted_genes[:i]) for i in range(num_genes_to_predict)]).transpose()

        # print iteration results
        print(scores[k])
        # if all_iterations == True:
        #     print(complete_scores[k])
        print("===============================")

    # compute average and std deviation of the metrics
    scores_avg = np.average(scores, axis=0)
    scores_std = np.std(scores, axis=0)

    if all_iterations == True:
        complete_scores_avg = np.average(complete_scores, axis=0)
        complete_scores_std = np.std(complete_scores, axis=0)

    print(f"scores_avg:\n{scores_avg}")
    print(f"score_std:\n{scores_std}")

    # if all_iterations == True:
        # print(f"complete_scores_avg:\n{complete_scores_avg}")
        # print(f"complete_score_std:\n{complete_scores_std}")

    # We are interested only in plotting them (see: plot_iteration_scores.py)
    if all_iterations == True:
        return complete_scores_avg

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
