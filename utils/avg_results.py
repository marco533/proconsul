import csv
import enum
import sys
from networkx import algorithms

import numpy as np
import pandas as pd

from data_utils import *

def print_latex(df, algorithm, outfile, validation='kfold'):
    '''
    Append to a txt file written in latex format for table
    to speed up the writing of the paper.
    '''
    values = df.values
    with open(outfile, 'a') as of:
        if validation == 'kfold':
            of.write("\\multicolumn{6}{|c|}{\\textbf{" + algorithm.upper() + "}} \\\\" + "\n")
            of.write("\\hline\n")
            of.write("& Top 50 & Top N/10 & Top N/4 & Top N/2 & Top N \\\\" + "\n")
            for i in range(values.shape[0]):
                if i == 0:
                    of.write(f"Precision & {values[i][0]} & {values[i][1]} & {values[i][2]} & {values[i][3]} & {values[i][4]} \\\\" + "\n")
                if i == 1:
                    of.write(f"Recall & {values[i][0]} & {values[i][1]} & {values[i][2]} & {values[i][3]} & {values[i][4]} \\\\" + "\n")
                if i == 2:
                    of.write(f"F1 & {values[i][0]} & {values[i][1]} & {values[i][2]} & {values[i][3]} & {values[i][4]} \\\\" + "\n")
                if i == 3:
                    of.write(f"NDCG & {values[i][0]} & {values[i][1]} & {values[i][2]} & {values[i][3]} & {values[i][4]} \\\\" + "\n")
            of.write("\\hline\n")

        if validation=='extended':
            of.write("\\multicolumn{3}{|c|}{\\textbf{" + algorithm.upper() + "}} \\\\" + "\n")
            of.write("\\hline\n")
            of.write("& Top 50 & Top N \\\\" + "\n")
            for i in range(values.shape[0]):
                if i == 0:
                    of.write(f"Precision & {values[i][0]} & {values[i][1]} \\\\" + "\n")
                if i == 1:
                    of.write(f"Recall & {values[i][0]} & {values[i][1]} \\\\" + "\n")
                if i == 2:
                    of.write(f"F1 & {values[i][0]} & {values[i][1]} \\\\" + "\n")
                if i == 3:
                    of.write(f"NDCG & {values[i][0]} & {values[i][1]} \\\\" + "\n")
            of.write("\\hline\n")

def get_algorithm_avg_results(disease_list, algorithm='diamond', validation='kfold', metric='f1'):
    '''
    Given the alogrithm the validation method and the metrics,
    Compute the average results of the algorith over the disease list.
    '''
    if validation == 'kfold':
        avg_result = np.zeros(5)
    else:
        avg_result = np.zeros(2)

    if validation == 'kfold':
        metric = 'avg_' + metric

    for disease in disease_list:
        # read csv file ad dataframe
        if validation == 'kfold':
            csv_file = f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv"
        else:
            csv_file = f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv"

        df = pd.read_csv(csv_file, index_col=0)  # first column as index

        # a ccumulate the scores of <algrorithm> on <disease> of <metric>
        avg_result = np.add(avg_result, get_score_from_dataframe(df, metric))

    # average the scores
    avg_result = np.divide(avg_result, len(disease_list))

    return avg_result

def get_algorithm_results(disease, algorithm='diamond', validation='kfold', metric='f1'):
    '''
    Given the alogrithm the validation method and the metrics,
    Compute the results of the algorith for the given disease.
    '''

    if validation == 'kfold':
        metric = 'avg_' + metric

    # read csv file ad dataframe
    if validation == 'kfold':
        csv_file = f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv"
    else:
        csv_file = f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv"

    df = pd.read_csv(csv_file, index_col=0)  # first column as index

    # get the results
    results = get_score_from_dataframe(df, metric)

    return results


def best_algorithm_per_disease(disease_file, output_file):
    '''
    Find the best algorithm for each disease, based on f1 score on top 50 genes
    and f1 score on top N.
    '''
    algorithm_list = ["diamond", "pdiamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR"]
    disease_list = get_diseases_from_file(disease_file)
    disease_num_genes_dict = get_num_genes_per_disease(disease_list)

    data_dict = {"Disease": [], "Num Genes": [],
                 "Best Alg top 50": [], "Best Alg top N": [],
                 "F1 score top 50": [], "F1 score top N": [] }

    for disease in disease_list:
        data_dict["Disease"].append(disease)
        data_dict["Num Genes"].append(disease_num_genes_dict[disease])

        # find best algorithm
        best_algorithm_top_50 = ""
        best_algorithm_top_N  = ""
        best_f1_top_50 = 0
        best_f1_top_N  = 0

        for algorithm in algorithm_list:
            # get the dataframe of <algorithm> on <disease>
            result_file = f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv"
            df = pd.read_csv(result_file, index_col=0)  # first column as index

            f1_top_50, f1_top_N = get_score_from_dataframe(df, "f1")

            if f1_top_50 > best_f1_top_50:
                best_algorithm_top_50 = algorithm
                best_f1_top_50 = f1_top_50

            if f1_top_N > best_f1_top_N:
                best_algorithm_top_N = algorithm
                best_f1_top_N = f1_top_N


        data_dict["Best Alg top 50"].append(best_algorithm_top_50)
        data_dict["Best Alg top N"].append(best_algorithm_top_N)
        data_dict["F1 score top 50"].append(best_f1_top_50)
        data_dict["F1 score top N"].append(best_f1_top_N)


    # write dataframe on a CSV file
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_file)


def avg_results_all_diseases(disease_file, by_mesh=False):
    '''
    Compute, for each algorithm, the average results for all diseases.
    If by_mesh=True, compute alsoe the average results by MeSH class.
    Save the results in output_file.
    '''
    algorithm_list = ["diamond", "pdiamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR"]
    disease_dict = create_disease_mesh_dictionary(disease_file)
    disease_list = get_diseases_from_file(disease_file)

    metrics = ["precision", "recall", "f1", "ndcg"]
    extended_sizes = ["top 50", "top N"]
    kfold_sizes = ["top 50", "top N/10", "top N/4", "top N/2", "top N"]

    kfold_outfile = "results/average_kfold_results_all_diseases.txt"
    extended_outfile = "results/average_extended_results_all_diseases.txt"

    kfold_outfile_latex = "results/latex_format/average_kfold_results_all_diseases_latex.txt"
    extended_outfile_latex = "results/latex_format/average_extended_results_all_diseases_latex.txt"

    # init output file
    with open(kfold_outfile, 'w') as kf, open(extended_outfile, 'w') as ef:
        kf.write("AVERAGE K-FOLD RESULTS FOR ALL DISEASES\n\n")
        ef.write("AVERAGE EXTENDED RESULTS FOR ALL DISEASES\n\n")

    # init latex file
    with open(kfold_outfile_latex, 'w') as kfl, open(extended_outfile_latex, 'w') as efl:
        kfl.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        kfl.write("\\hline\n")
        kfl.write("\\multicolumn{6}{|c|}{\\texttt{5-Fold Cross Validation}} \\\\" + "\n")
        kfl.write("\hline\n")

        efl.write("\\begin{tabular}{|c|c|c|}\n")
        efl.write("\\hline\n")
        efl.write("\\multicolumn{3}{|c|}{\\texttt{Extended Validation}} \\\\" + "\n")
        efl.write("\hline\n")

    # all diseases average
    for algorithm in algorithm_list:
        extended_results = np.zeros((len(disease_list), len(metrics), len(extended_sizes)))
        kfold_results = np.zeros((len(disease_list), len(metrics), len(kfold_sizes)))
        kfold_stds = np.zeros((len(disease_list), len(metrics), len(kfold_sizes)))

        for idx, disease in enumerate(disease_list):
            # import csv files with results
            kfold_df = pd.read_csv(f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv", index_col=0)
            extended_df = pd.read_csv(f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv", index_col=0)

            # get extended results
            extended_results[idx] = np.array(extended_df.values)

            # get kfold results
            kfold_results[idx], kfold_stds[idx] = split_result_std(kfold_df)

        # average results
        avg_extended_results = np.average(extended_results, axis=0)
        extended_stds = np.std(extended_results, axis=0)

        avg_kfold_results = np.average(kfold_results, axis=0)
        avg_kfold_stds = np.sqrt(np.average(np.square(kfold_results), axis=0))

        # save the average results in a table
        avg_kfold_table = np.empty((len(metrics), len(kfold_sizes)), dtype=object)
        for i in range(len(metrics)):
            for j in range(len(kfold_sizes)):
                avg_kfold_table[i][j] = str(round(avg_kfold_results[i][j], 3)) + " +/- " + str(round(avg_kfold_stds[i][j], 3))

        avg_extended_table = np.empty((len(metrics), len(extended_sizes)), dtype=object)
        for i in range(len(metrics)):
            for j in range(len(extended_sizes)):
                avg_extended_table[i][j] = str(round(avg_extended_results[i][j], 3)) + " +/- " + str(round(extended_stds[i][j], 3))

        # convert the tables in datafram
        avg_kfold_df = pd.DataFrame(avg_kfold_table, metrics, kfold_sizes)
        avg_extended_df = pd.DataFrame(avg_extended_table, metrics, extended_sizes)


        # append the dataframe on a file
        with open(kfold_outfile, 'a') as kf, open(extended_outfile, 'a') as ef:
            # K-FOLD
            kf.write(f"----------------------------------------------------------------------------------------------\n")
            kf.write(f"                             {algorithm.upper()} SCORE\n")
            kf.write(f"----------------------------------------------------------------------------------------------\n")
            kf.write(avg_kfold_df.to_string())
            kf.write(f"\n\n")

            # EXTENDED
            ef.write(f"--------------------------------------------\n")
            ef.write(f"         {algorithm.upper()} SCORE\n")
            ef.write(f"--------------------------------------------\n")
            ef.write(avg_extended_df.to_string())
            ef.write(f"\n\n")

        # save table in LaTex formatting
        print_latex(avg_kfold_df, algorithm, kfold_outfile_latex, validation='kfold')
        print_latex(avg_extended_df, algorithm, extended_outfile_latex, validation='extended')

    # close latex file
    with open(kfold_outfile_latex, 'a') as kfl, open(extended_outfile_latex, 'a') as efl:
        kfl.write("\\end{tabular}\n")

        efl.write("\\end{tabular}\n")


    if by_mesh==True:
        kfold_outfile = "results/average_kfold_results_by_mesh_class.txt"
        extended_outfile = "results/average_extended_results_by_mesh_class.txt"

        kfold_outfile_latex = "results/latex_format/average_kfold_results_by_mesh_class_latex.txt"
        extended_outfile_latex = "results/latex_format/average_extended_results_by_mesh_class_latex.txt"

        # init output file
        with open(kfold_outfile, 'w') as kf, open(extended_outfile, 'w') as ef:
            kf.write("AVERAGE K-FOLD RESULTS BY MESH CLASS\n\n")
            ef.write("AVERAGE EXTENDED RESULTS BY MESH CLASS\n\n")

        # init latex file
        with open(kfold_outfile_latex, 'w') as kfl, open(extended_outfile_latex, 'w') as efl:
            kfl.write("\\begin{longtable}{|c|c|c|c|c|c|c|c|}\n")
            kfl.write("\\hline\n")
            kfl.write("MeSH Class & Algorithm & & Top 50 & Top N/10 & Top N/4 & Top N/2 & Top N \\\\" + "\n")
            kfl.write("\hline\n")

            efl.write("\\begin{longtable}{|c|c|c|}\n")
            efl.write("\\hline\n")
            kfl.write("MeSH Class & Algorithm & & Top 50 & Top N \\\\" + "\n")
            efl.write("\hline\n")

        classes = disease_dict.keys()

        for c in classes:
            # get diseases in class c
            disease_sublist = disease_dict[c]

            with open(kfold_outfile, 'a') as kf, open(extended_outfile, 'a') as ef:
                kf.write(f"# ======================== #\n")
                kf.write(f"#      MESH CLASS {c}      #\n")
                kf.write(f"# ======================== #\n\n")

                ef.write(f"# ======================== #\n")
                ef.write(f"#      MESH CLASS {c}      #\n")
                ef.write(f"# ======================== #\n\n")

            # mesh class disease average
            for algorithm in algorithm_list:
                extended_results = np.zeros((len(disease_list), len(metrics), len(extended_sizes)))
                kfold_results = np.zeros((len(disease_list), len(metrics), len(kfold_sizes)))
                kfold_stds = np.zeros((len(disease_list), len(metrics), len(kfold_sizes)))

                for idx, disease in enumerate(disease_sublist):
                    # import csv files with results
                    kfold_df = pd.read_csv(f"results/kfold/csv_format/average/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_average_kfold.csv", index_col=0)
                    extended_df = pd.read_csv(f"results/extended/csv_format/{string_to_filename(algorithm)}_on_{string_to_filename(disease)}_extended_validation.csv", index_col=0)

                    # get extended results
                    extended_results[idx] = np.array(extended_df.values)

                    # get kfold results
                    kfold_results[idx], kfold_stds[idx] = split_result_std(kfold_df)

                # average results
                avg_extended_results = np.average(extended_results, axis=0)
                extended_stds = np.std(extended_results, axis=0)

                avg_kfold_results = np.average(kfold_results, axis=0)
                avg_kfold_stds = np.sqrt(np.average(np.square(kfold_results), axis=0))

                # save the average results in a table
                avg_kfold_table = np.empty((len(metrics), len(kfold_sizes)), dtype=object)
                for i in range(len(metrics)):
                    for j in range(len(kfold_sizes)):
                        avg_kfold_table[i][j] = str(round(avg_kfold_results[i][j], 3)) + " +/- " + str(round(avg_kfold_stds[i][j],3))

                avg_extended_table = np.empty((len(metrics), len(extended_sizes)), dtype=object)
                for i in range(len(metrics)):
                    for j in range(len(extended_sizes)):
                        avg_extended_table[i][j] = str(round(avg_extended_results[i][j], 3)) + " +/- " + str(round(extended_stds[i][j], 3))

                # convert the tables in datafram
                avg_kfold_df = pd.DataFrame(avg_kfold_table, metrics, kfold_sizes)
                avg_extended_df = pd.DataFrame(avg_extended_table, metrics, extended_sizes)


                # append the dataframe on a file
                with open(kfold_outfile, 'a') as kf, open(extended_outfile, 'a') as ef:
                    # K-FOLD
                    kf.write(f"---------------------------------------\n")
                    kf.write(f"\t{algorithm.upper()} SCORE\n")
                    kf.write(f"---------------------------------------\n")
                    kf.write(f"{avg_kfold_df.to_string()}")
                    kf.write(f"\n\n")

                    # EXTENDED
                    ef.write(f"---------------------------------------\n")
                    ef.write(f"\t{algorithm.upper()} SCORE\n")
                    ef.write(f"---------------------------------------\n")
                    ef.write(f"{avg_extended_df.to_string()}")
                    ef.write(f"\n\n")





if __name__ == "__main__":
    best_algorithm_per_disease("data/disease_file.txt", "results/best_algorithm_per_disease.csv")
    avg_results_all_diseases("data/disease_file.txt", by_mesh=True)