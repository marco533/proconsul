import csv
import enum
import sys
from unittest import result

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize

from utils.data_utils import *
from avg_results import get_algorithm_avg_results, get_algorithm_results

####################################
#   K-FOLD ITERATIONS COMPARISON   #
####################################

def plot_kfold_comparison(algorithm1, algorithm2, disease_list, metric='f1', prediction_size="top 50", K=5):
    print(f"{algorithm1}_vs_{algorithm2}_{metric}_{string_to_filename(prediction_size)}")

    # get kfold result for each disease
    alg1_result = {}
    alg2_result = {}
    for disease in disease_list:
        # algorithm 1
        csv_file = f"results/kfold/csv_format/iterations/{string_to_filename(algorithm1)}_on_{string_to_filename(disease)}_kfold_iterations.csv"
        df = pd.read_csv(csv_file, index_col=0, header=[0,1])
        alg1_result[disease] = get_score_from_dataframe(df, metric, keys=[prediction_size, metric], multi_index=True)

        # algorithm 2
        csv_file = f"results/kfold/csv_format/iterations/{string_to_filename(algorithm2)}_on_{string_to_filename(disease)}_kfold_iterations.csv"
        df = pd.read_csv(csv_file, index_col=0, header=[0,1])
        alg2_result[disease] = get_score_from_dataframe(df, metric, keys=[prediction_size, metric], multi_index=True)

    # plot result
    iteration_labels = [f"iter {i}" for i in range(1, K+1)]
    results = np.zeros((len(disease_list), len(iteration_labels)))
    # print(results)

    for i, disease in enumerate(disease_list):
        for j, _ in enumerate(iteration_labels):
            if alg1_result[disease][j] >= alg2_result[disease][j]:
                results[i][j] = alg1_result[disease][j]
            else:
                results[i][j] = -alg2_result[disease][j]

    # print(results)


    # define the colormap
    cmap = plt.get_cmap('PuOr')

    # init plot
    fig, ax = plt.subplots(1,1)
    img = plt.imshow(results, interpolation='none', aspect='auto', cmap=cmap, norm=Normalize(vmin=-0.1, vmax=0.1))

    # set lables
    ax.set_yticks(range(0,len(disease_list)))
    ax.set_xticks(range(0,len(iteration_labels)))

    ax.set_xticklabels(iteration_labels)
    ax.set_yticklabels(disease_list, fontsize=6)

    # save plot
    fig.colorbar(img)
    fig.suptitle(f"{algorithm1} VS {algorithm2} [{metric} - {prediction_size}]")
    plt.text(4, -2, f"{algorithm1} > {algorithm2}")
    plt.text(4, 49, f"{algorithm2} > {algorithm1}")
    plt.savefig(f"plot/kfold/{metric}/{algorithm1}_vs_{algorithm2}_{metric}_{string_to_filename(prediction_size)}.png", bbox_inches="tight")
    plt.close('all')


##################################################
# AVERAGE SCORE OF THE ALGORITHMS PER INPUT SIZE #
##################################################

def plot_average_comparison(size_labels, algorithm_names, disease_dict, metric='f1', validation='kfold', by_class=False):
    '''
    Plot the average score of all algorithms.
    Parameters:
        - size_labels:      list of the size of predicted genes (e.g. top 50, top N, top N/2, ...)
        - algorithm_names:  list with the names of the algorithm of which we want to compte the average score
        - disease_dict:     a dictonary that store the MeSH classes and the currespondent diseases
        - metric:           average the scores of this metric (e.g. 'precision', 'recall', 'f1', 'ndcg')
        - validation:       if 'kfold' average the score of K-Fold Cross Validation, if 'extended' average
                            the score of extended validation.
        - by_class:         if False -> compute the average score over all the diseases.
                            if True  -> compute the average score over all the diseases and the
                                        averages by MeSH Classes.

    Plot the results in the plot folder
    And save the averages in results folder as CSV file.

    Returns: DataFrame of the overall averages & Dataframe of the MeSH Class averages
    '''

    # ================== #
    #   Global Average   #
    # ================== #

    print(f"plot average_{fix_metric_name(metric)}_{validation}")

    # get the list of all the diseases
    disease_list_of_list = disease_dict.values() # get dictionary values
    disease_list = [disease for sublist in disease_list_of_list for disease in sublist] # flat the list
    disease_list = list(set(disease_list)) # remove duplicate disease
    # print(len(disease_list))

    # compute for each algorithm the average of the metric <metric> over all the diseases
    results_dict = {}

    for algorithm in algorithm_names:
        algorithm_score = np.zeros(len(size_labels))

        for disease in disease_list:
            # read csv file ad dataframe
            if validation == 'kfold':
                csv_file = f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv"
            else:
                csv_file = f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv"
            df = pd.read_csv(csv_file, index_col=0)  # first column as index

            # accumulate the scores of <algrorithm> on <disease> of <metric>
            algorithm_score = np.add(algorithm_score, get_score_from_dataframe(df, metric))

        # average the scores
        algorithm_score = np.divide(algorithm_score, len(disease_list))

        # add the averaged result to the dictionary
        results_dict[algorithm] = algorithm_score

    # create a plottable dataframe with the average results of the algorithms for each size
    average_df = pd.DataFrame(results_dict, size_labels)

    # plot dataframe
    average_df.plot.barh()
    plt.title(f'Average {fix_metric_name(metric)} by algorithm and input size [{validation}]')
    plt.savefig(f"plot/average/average_{fix_metric_name(metric)}_{validation}.png", bbox_inches="tight")
    plt.close('all')

    # ================== #
    #    MeSH Average    #
    # ================== #
    if by_class == True:

        # get classes
        classes = disease_dict.keys()

        # for each class compute the average and save on a file
        for c in classes:
            class_result_dict = {}

            print(f"plot average_{fix_metric_name(metric)}_on_mesh_class_{c}_{validation}")
            diseases_in_c = disease_dict[c]

            for algorithm in algorithm_names:
                algorithm_score = np.zeros(len(size_labels))

                for disease in diseases_in_c:
                    # read csv file ad dataframe
                    if validation == 'kfold':
                        csv_file = f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv"
                    else:
                        csv_file = f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv"

                    df = pd.read_csv(csv_file, index_col=0)  # first column as index

                    # accumulate the scores of <algrorithm> on <disease> of <metric>
                    algorithm_score = np.add(algorithm_score, get_score_from_dataframe(df, metric))

                # average the scores
                algorithm_score = np.divide(algorithm_score, len(disease_list))

                # add the averaged result to the dictionary
                class_result_dict[algorithm] = algorithm_score

            # create a plottable dataframe with the average results for each class of the algorithms for each size
            mesh_df = pd.DataFrame(class_result_dict, size_labels)

            # plot dataframe
            mesh_df.plot.barh()
            plt.title(f'Average {fix_metric_name(metric)} on MeSH Class {c} [{validation}]')
            plt.savefig(f"plot/average/by_class/average_{fix_metric_name(metric)}_on_mesh_class_{c}_{validation}.png", bbox_inches="tight")
            plt.close('all')


def plot_success_rate_by_class(disease_file, validation='kfold', metric='f1', size='top 50'):
    algorithm_names = ["diamond", "prob_diamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR"]

    disease_dict = create_disease_mesh_dictionary(disease_file)
    classes = disease_dict.keys()
    results = {}

    for c in classes:
        # init the dictionary
        results[c] = []
        for algorithm in algorithm_names:
            avg_results = get_algorithm_avg_results(disease_dict[c], algorithm=algorithm, validation=validation, metric=metric)

            if validation == 'kfold':

                # exclude moses and mcl result with size > 50
                if size != "top 50" and algorithm in ["moses", "markov_clustering"]:
                    avg_results = [0, 0, 0, 0, 0]

                # take the specific metric
                if size == "top 50":
                    results[c].append(avg_results[0])
                elif size == "top N/10":
                    results[c].append(avg_results[1])
                elif size == "top N/4":
                    results[c].append(avg_results[2])
                elif size == "top N/10":
                    results[c].append(avg_results[3])
                else:
                    results[c].append(avg_results[4])
            if validation == 'extended':

                # exclude moses and mcl result with size > 50
                if size != "top 50" and algorithm in ["moses", "markov_clustering"]:
                    avg_results = [0, 0, 0, 0, 0]

                if size == "top 50":
                    results[c].append(avg_results[0])
                else:
                    results[c].append(avg_results[1])

    labels = list(results.keys())
    data = np.array(list(results.values()))

    # normalize data
    norm_data = []
    for i in range(data.shape[0]):
        values = data[i]

        # norm values
        norm_values = ((values - values.min()) / (values - values.min()).sum())
        # round values to the 2nd decimal
        round_values = np.round(norm_values,3)

        norm_data.append(round_values * 100 )

    # overwrite data with the normalized ones
    data = np.array(norm_data)

    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(algorithm_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncol=len(algorithm_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.title(f'Algorithms success rate by class - {size} predicted genes - {validation.upper()} VALIDATION', y=1.08)
    plt.savefig(f"plot/algorithms_success_rate_by_class_{validation}_{string_to_filename(size)}.png", bbox_inches="tight")
    plt.close('all')


def plot_algorithm_comparison_by_num_genes(disease_file, validation='kfold', metric='f1', size='top 50'):
    '''
    Plot how the algorithm score change with the number of disease genes.
    '''

    algorithm_names = ["diamond", "prob_diamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR"]

    disease_list = get_diseases_from_file(disease_file)

    # get the number of genes for each disease
    disease_num_genes_dict = get_num_genes_per_disease(disease_list)

    # sort the dictionary by the number of genes
    disease_num_genes_dict = dict(sorted(disease_num_genes_dict.items(), key=lambda item: item[1]))
    ordered_disease_list = disease_num_genes_dict.keys()

    # get the <metric> score for the <top 50> genes of each algorithm and each disease
    # IMPORTANT: Iterate over the dictionary keys because they are sorted
    for algorithm in algorithm_names:
        # init data
        x = []
        y = []
        for disease in ordered_disease_list:
            if validation == 'kfold':
                all_sizes_results = get_algorithm_results(disease, algorithm=algorithm, validation=validation, metric=metric)

                # exclude moses and mcl result with size > 50
                if size != "top 50" and algorithm in ["moses", "markov_clustering"]:
                    all_sizes_results = [0, 0, 0, 0, 0]

                if size == "top 50":
                    result = all_sizes_results[0]
                elif size == "top N/10":
                    result = all_sizes_results[1]
                elif size == "top N/4":
                    result = all_sizes_results[2]
                elif size == "top N/2":
                    result = all_sizes_results[3]
                else:
                    result = all_sizes_results[4]
            if validation == "extended":
                all_sizes_results = get_algorithm_results(disease, algorithm=algorithm, validation=validation, metric=metric)

                # exclude moses and mcl result with size > 50
                if size != "top 50" and algorithm in ["moses", "markov_clustering"]:
                    all_sizes_results = [0, 0]

                if size == "top 50":
                    result = all_sizes_results[0]
                else:
                    result = all_sizes_results[1]

            # print(result)
            # sys.exit(0)
            x.append(disease_num_genes_dict[disease])
            y.append(result)

        # add the <algorithm> plot
        plt.plot(x, y, label=algorithm)

    # print(f"x data: {x}")
    # print(f"y data: {y}")
    # sys.exit(0)
    # plot legend and save as png
    plt.legend()
    plt.title(f"Algorithm score by number of seed genes - {metric} Score | {size} Predicted Genes | {validation.upper()} Validation")
    plt.savefig(f"plot/algorithms_score_by_num_seed_genes_{metric}_{string_to_filename(size)}_{validation}.png", bbox_inches="tight")
    plt.close('all')

############
#   MAIN   #
############

if __name__ == "__main__":
    # labels
    input_size_labels_kfold     = ["top 50", "top N/10", "top N/4", "top N/2", "top N"]
    input_size_labels_extended  = ["top 50", "top N"]
    average_metric_labels       = ["avg_precision", "avg_recall", "avg_f1", "avg_ndcg"]
    metric_labels               = ["precision", "recall", "f1", "ndcg"]
    algorithm_names             = ["diamond", "prob_diamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR"]
    disease_dict                = create_disease_mesh_dictionary("data/disease_file.txt")

    # ====================== #
    #   Average Comparsion   #
    # ====================== #
    print("Algorithm comparison by number of genes")

    plot_algorithm_comparison_by_num_genes("data/disease_file.txt", validation='kfold', metric='f1', size='top 50')
    plot_algorithm_comparison_by_num_genes("data/disease_file.txt", validation='extended', metric='f1', size='top 50')
    plot_algorithm_comparison_by_num_genes("data/disease_file.txt", validation='kfold', metric='f1', size='top N')
    plot_algorithm_comparison_by_num_genes("data/disease_file.txt", validation='extended', metric='f1', size='top N')

    print("\n============================================================\n")

    print("Success rate by class")
    plot_success_rate_by_class("data/disease_file.txt", validation='kfold', metric='f1', size='top 50')
    plot_success_rate_by_class("data/disease_file.txt", validation='extended', metric='f1', size='top 50')
    plot_success_rate_by_class("data/disease_file.txt", validation='kfold', metric='f1', size='top N')
    plot_success_rate_by_class("data/disease_file.txt", validation='extended', metric='f1', size='top N')

    print("\n============================================================\n")


    print("Average comparsion")

    for metric in average_metric_labels:
        plot_average_comparison(input_size_labels_kfold,
                                algorithm_names,
                                disease_dict,
                                metric=metric,
                                validation='kfold',
                                by_class = True)

    for metric in metric_labels:
        plot_average_comparison(input_size_labels_extended,
                                algorithm_names,
                                disease_dict,
                                metric=metric,
                                validation='extended',
                                by_class = True)

    print("\n============================================================\n")

    # ======================= #
    #    K-Fold Comparsion    #
    # ======================= #
    print("K-Fold Comparison")

    # retrieve disease names
    disease_names = get_diseases_from_file("data/disease_file.txt")

    for algorithm in algorithm_names:
        for metric in metric_labels:
            for size in input_size_labels_kfold:
                if algorithm == "diamond":
                    continue
                plot_kfold_comparison("diamond", algorithm, disease_names, metric=metric, prediction_size=size)

    print("\n============================================================\n")


