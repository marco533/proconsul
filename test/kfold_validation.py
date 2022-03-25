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


# cross validation
def k_fold_cross_validation(network, seed_genes, algorithm, disease_name, K=5, diffusion_time=0.005, num_iters_prob_diamond=10):
    '''
    Performs the K-Folf Cross Validation over all the disease genes.
    Input:
        - network:       the LCC of our Protein-Protein Interactions
        - disease_genes: the list of disease genes
        - algorithm:     the algorithm to test
        - K:             The number of folds (default: 5)
    Output:
        List[List] of the performance metrics on the algorithm
    '''

    # get all genes in the network
    all_genes = list(network.nodes)

    # write connections in a file
    interactome = "data/HHI_LCC.txt"
    write_edges_in_file(network, interactome)

    # split  list in K equal parts
    splitted_disease_genes = split_list(seed_genes, K)
    num_disease_genes = len(seed_genes)

    # how many genes predict
    num_genes_to_predict = 200

    # prepare the files for the computation
    test_set_file = "tmp/test_set.txt"
    training_set_file = "tmp/training_set.txt"

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

        # write genes in a file
        with open(test_set_file, 'w') as fp, open(training_set_file, 'w') as ft:
            for gene in test_genes:
                fp.write(gene + "\n")
            for gene in training_genes:
                ft.write(gene + "\n")

        # run algorithm
        output_file = f"tmp/{algorithm}_output.txt"

        # if the algorithm doesn't return a ranking set this flag False
        ranking_flag = True

        if algorithm == "diamond":
            input_list = ["DIAMOnD.py", interactome, training_set_file, num_genes_to_predict, 1, output_file]
            added_nodes = run_diamond(input_list)   # take only firs elements of each sublist
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "prob_diamond":
            input_list = ["prob_diamond.py", interactome, training_set_file, num_genes_to_predict, 1, output_file, num_iters_prob_diamond]
            added_nodes = run_prob_diamond(input_list)
            predicted_genes = [item[0] for item in added_nodes]

        elif algorithm == "heat_diffusion":
            predicted_genes = run_heat_diffusion(network, training_genes, n_positions=num_genes_to_predict, diffusion_time=diffusion_time)

        else:
            print("  ERROR: No valid algorithm.    ")
            print("  Choose one of the following:  ")
            print("    - diamond                   ")
            print("    - prob_diamond              ")
            print("    - heat_diffusion            ")
            sys.exit(0)

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

    # save it as csv file
    if algorithm == "diamond":
        csv_file = f"results/kfold/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_kfold.csv"

    if algorithm == "prob_diamond":
        csv_file = f"results/kfold/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_kfold_{num_iters_prob_diamond}_iters.csv"

    if algorithm == "heat_diffusion":
        csv_file = f"results/kfold/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_kfold_diff_time_{diffusion_time}.csv"

    result_df.to_csv(csv_file)
