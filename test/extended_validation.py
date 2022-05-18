import sys

import networkx as nx
import numpy as np
import pandas as pd
from algorithms.diamond import DIAMOnD
from algorithms.pdiamond import pDIAMOnD
from algorithms.pdiamond_rank import pDIAMOnD_rank
from algorithms.pdiamond_temp import pDIAMOnD_temp
from algorithms.pdiamond_topk import pDIAMOnD_topk
from algorithms.pdiamond_all import pDIAMOnD_all
from algorithms.heat_diffusion import run_heat_diffusion
from utils.network_utils import *
from utils.metrics_utils import *
from utils.data_utils import *


def extended_validation(network, seed_genes, test_genes, algorithm, disease_name, diffusion_time=0.005, num_iters_pdiamond=10):
    '''
    Perform an extended validation using all the seed genes of the disease
    and test the predicted genes with all the genes in
    all_disease_gene_associations.tsv
    '''

    # get all genes in the network
    all_genes = list(network.nodes)

    # get number of seed genes
    num_disease_genes = len(seed_genes)

    # how many genes predict
    num_genes_to_predict = 200

    # Extended Validation:
    # compute the score for each algorithm using the top 25, top 50, top 100 and top 200 predicted genes
    print(f"Extended validation of {algorithm.upper()} on {disease_name.upper()}...", end="\n\n")

    # output file for diamond-like algorithms
    outfile = f"tmp/{algorithm}_output.txt"

    predicted_genes = []

    if algorithm == "diamond":
        added_nodes = DIAMOnD(network, seed_genes, num_genes_to_predict, 1, outfile=outfile)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond":
        added_nodes = pDIAMOnD(network, seed_genes, num_genes_to_predict, 1, outfile=outfile, max_num_iterations=num_iters_pdiamond)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond_rank":
        added_nodes = pDIAMOnD_rank(network, seed_genes, num_genes_to_predict, 1, outfile=outfile, max_num_iterations=num_iters_pdiamond)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond_temp":
        added_nodes = pDIAMOnD_temp(network, seed_genes, num_genes_to_predict, 1, outfile=outfile, max_num_iterations=num_iters_pdiamond, T_start=0.1, T_step=0.1)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond_topk":
        added_nodes = pDIAMOnD_topk(network, seed_genes, num_genes_to_predict, 1, outfile=outfile, max_num_iterations=num_iters_pdiamond)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "pdiamond_all":
        added_nodes = pDIAMOnD_all(network, seed_genes, num_genes_to_predict, 1, outfile=outfile, max_num_iterations=num_iters_pdiamond, T_start=0.1, T_step=0.1)
        predicted_genes = [item[0] for item in added_nodes]

    elif algorithm == "heat_diffusion":
        predicted_genes = run_heat_diffusion(network, seed_genes, n_positions=num_genes_to_predict, diffusion_time=diffusion_time)


    else:
        print("  ERROR: No valid algorithm.    ")
        print("  Choose one of the following:  ")
        print("    - diamond                   ")
        print("    - pdiamond                  ")
        print("    - pdiamond_rank             ")
        print("    - pdiamond_temp             ")
        print("    - pdiamond_topk             ")
        print("    - pdiamond_all              ")
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
    csv_filename = f"results/extended/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease_name)}_-_extended.csv"

    # add additional information in the filename
    if "pdiamond" in algorithm:
        csv_filename = csv_filename.replace(".csv", f"_-_{num_iters_pdiamond}_iterations.csv")

    if algorithm == "heat_diffusion":
        csv_filename = csv_filename.replace(".csv", f"_-_diffusion_time_{diffusion_time}.csv")

    result_df.to_csv(csv_filename)
