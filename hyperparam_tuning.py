import sys
import networkx as nx

from test.kfold_validation import k_fold_cross_validation
from utils.data_utils import *
from utils.network_utils import *

# =============================================== #
# Find best number of iterations for Prob DIMAOnD #
# =============================================== #

if __name__ == "__main__":

    #================#
    # CREATE NETWORK #
    #================#

    # select the human-human interactions from biogrid
    biogrid_file = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df = select_hhi_only(biogrid_file)

    # create the hhi from the filtered data frame
    hhi = nx.from_pandas_edgelist(hhi_df,
                                  source = "Official Symbol Interactor A",
                                  target = "Official Symbol Interactor B",
                                  create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges
    print("Network Info:")
    print(nx.info(hhi), end="\n\n")

    # remove self loops
    self_loop_edges = list(nx.selfloop_edges(hhi))
    hhi.remove_edges_from(self_loop_edges)
    print("After removing self-loops:")
    print(nx.info(hhi), end="\n\n")

    # isolate the largest connected component
    LCC_hhi = isolate_LCC(hhi)

    print("Isolating the LCC:")
    print(nx.info(LCC_hhi), end="\n\n")

    # ================== #
    #  Get disease info  #
    # ================== #

    gda_filename = "data/curated_gene_disease_associations.tsv"

    # get disease genes from curated GDA
    disease_genes = get_disease_genes_from_gda(gda_filename, "Psoriasis")

    # ============================ #
    #  Find best iterations value  #
    # ============================ #

    results_dict = {"Num iterations": [], "F1 top 50": [], "F1 top 100": [], "F1 top 200": [], "F1 top N": []}

    best_num_iters = 0
    best_f1_top_100 = 0
    for i in range(1,21):
        k_fold_cross_validation(LCC_hhi, disease_genes, "prob_diamond", "Psoriasis", K=5, num_iters_prob_diamond=i)

        # read the results DataFrame
        results_df = pd.read_csv("results/kfold/prob_diamond_on_psoriasis_kfold.csv")

        # append f1 results to the dictionary
        results_dict["Num iterations"] .append(i)
        results_dict["F1 top 50"] .append(results_df.at["f1", "Top 50"][0])
        results_dict["F1 top 100"] .append(results_df.at["f1", "Top 100"][0])
        results_dict["F1 top 200"] .append(results_df.at["f1", "Top 200"][0])
        results_dict["F1 top N"] .append(results_df.at["f1", "Top N"][0])

        # find best f1 top 100 score
        f1_top_100 = results_df.at["f1", "Top 100"][0]
        if f1_top_100 > best_f1_top_100:
            best_num_iters = i
            best_f1_top_100 = f1_top_100

    # save the dictionary in a csv file
    print(results_dict)
    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv("results/prob_diamond_score_per_num_of_iterations.csv", float_format='%.3f')

    print(f"Best num iterations for Prob DIAMOnD is: {best_num_iters}, with an f1 score of {best_f1_top_100}")