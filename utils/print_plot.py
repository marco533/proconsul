import csv
import enum
import re
import sys
import argparse
from unittest import result

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from sklearn import metrics

from data_utils import *
from avg_results import get_algorithm_avg_results, get_algorithm_results
from get_disease_LCC import get_disease_LCC, select_hhi_only

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

def plot_scores_by_lcc_size(interactome, disease_list, algorithm_list, validation='kfold', metric='f1', output_size='Top 100'):
    '''
    Plot the score of all algorithms in algorithm_list
    by the LCC size of the diseases in disease file.
    '''

    # Order disease list by LCC size
    diseases_by_lcc_size = {}
    for disease in disease_list:
        # Get disease LCC
        disease_LCC = get_disease_LCC(interactome,
                                      disease,
                                      from_curated=True if validation=='kfold' else False)

        # Add the value to the dictionary
        diseases_by_lcc_size[disease] = len(disease_LCC)

    # Sort the dictionary by the LCC size in ascending order
    diseases_by_lcc_size = dict(sorted(diseases_by_lcc_size.items(), key=lambda item: item[1]))

    # Get the sorted disease list
    sorted_disease_list = diseases_by_lcc_size.keys()

    # Get algorithm score
    for algorithm in algorithm_list:
        LCC_sizes = []
        scores = []

        for disease in sorted_disease_list:
            # Read score from saved CSV
            score_file = f"results/{validation}/{algorithm}/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_{validation}.csv"
            score_df = pd.read_csv(score_file, index_col=0)

            if validation == 'kfold':
                score = score_df.at[metric, output_size]

                if type(score) == str:
                    score = score.replace("(", "")
                    score = score.replace(")", "")
                    score = float(score.split(", ")[0])

            else:
                score = score_df.at[metric, output_size]

            # Append score and LCC size
            LCC_sizes.append(diseases_by_lcc_size[disease])
            scores.append(score)

        # print(f"LCC sizes: {LCC_sizes}")
        # print(f"Scores: {scores}")
        # Add the algorithm score to the plot
        plt.plot(LCC_sizes, scores, label=algorithm)

    # plot legend and save as png
    plt.legend()
    plt.title(f"Scores by LCC size - Metric: {metric.upper()} | Predicted genes: {output_size} | Validation: {validation.upper()}")
    plt.savefig(f"plots/scores_by_lcc_size/{string_to_filename(output_size)}_{metric}_{validation}.png", bbox_inches="tight")
    plt.close('all')

# =============== #
#  INPUT READING  #
# =============== #

def print_usage():
    print(' ')
    print('        usage: python3 project.py --algorithm --validation --disease_file')
    print('        -----------------------------------------------------------------')
    print('        disease_file     : Position of a txt file containig a disease name for each line')
    print('                           (default: "data/disease_file.txt"')
    print('        algorithm        : Algorithm to test. It can be "diamond", "prob_diamond or "all".')
    print('                           If all, run both the algorithms. (default: all')
    print('        validation       : type of validation on which test the algorithms. It can be')
    print('                           "kfold", "extended" or "all".')
    print('                           If all, perform both the validations. (default: all')
    print('        output_size      : Number of predicted genes to use for plotting the results.')
    print('                           "Top 50", "Top 100", "Top 200", "Top N" or "all".')
    print('                           If all, use all the sizes (default: all).')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithm and validation')
    parser.add_argument('--algorithm', type=str, default='all',
                    help='Algorithm to test. (default: all)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Position to disease file (default: "data/disease_file.txt)')
    parser.add_argument('--output_size', type=str, default='all',
                        help='Number of predicted genes to use.')
    return parser.parse_args()

def read_terminal_input(args):
    '''
    Read the arguments passed by command line.
    '''

    def read_disease_file(disease_file):
        '''
        Read the disease file and return a list of diseases.
        The file MUST HAVE only a desease name for each line.
        '''
        disease_list = []
        with open(disease_file, 'r') as df:

            for line in df:
                if line[0] == "#":  # skip commented lines
                    continue
                disease_list.append(line.replace("\n",""))

        return disease_list

    # read the parsed values
    algorithm       = args.algorithm
    validation      = args.validation
    output_size     = args.output_size
    disease_file    = args.disease_file

    print('')
    print(f"============================")

    print(f"algorithm: {algorithm}")
    print(f"validation: {validation}")
    print(f"output_size: {output_size}")
    print(f"disease_file: {disease_file}")

    # get disease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if list is empty fill it with default diseases
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    print(f"============================")
    print('')

    # check if is a valid algorithm
    if algorithm not in ["diamond", "prob_diamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR", "all"]:
        print(f"ERROR: {algorithm} is no valid algorithm!")
        print_usage()
        sys.exit(0)

    # check if is a valid validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(0)

    if output_size not in ["Top 50", "Top 100", "Top 200", "Top N", "all"]:
        print(f"ERROR: {output_size} is no valid matric!")
        print_usage()
        sys.exit(0)

    return disease_list, algorithm, validation, output_size

# ======== #
#   MAIN   #
# ======== #

if __name__ == "__main__":

    # Read input
    args = parse_args()
    disease_list, algorithm, validation, output_size = read_terminal_input(args)

    if algorithm == 'all':
        algorithm_list = ['diamond', 'prob_diamond']
    else:
        algorithm_list = [algorithm]

    if validation == 'all':
        validations = ['kfold', 'extended']
    else:
        validations = [validation]

    if  output_size == 'all':
        output_sizes = ["Top 50", "Top 100", "Top 200", "Top N"]
    else:
        output_sizes = [output_size]


    # Build the interactome DataFrame
    biogrid_file = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df = select_hhi_only(biogrid_file)

    # Plot F1 and NDCG scores by LCC size
    for validation in validations:
        for size in output_sizes:
            print(f"Plotting F1 score of {validation} on {size} genes")
            plot_scores_by_lcc_size(hhi_df, disease_list, algorithm_list, validation=validation, metric='f1', output_size=size)
            print(f"Plotting NDCG score of {validation} on {size} genes")
            plot_scores_by_lcc_size(hhi_df, disease_list, algorithm_list, validation=validation, metric='ndcg', output_size=size)