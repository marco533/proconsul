
import sys

import numpy as np
import pandas as pd

from utils.data_utils import string_to_filename


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

def get_score(algorithm=None, disease=None, database=None, validation=None, K=None, diffusion_time=None, n_iters=None, temp=None, top_p=None, top_k=None):

    # Get the relative path to the algorithm score
    score_path = f"results/{database}/{validation}/{algorithm}/{algorithm}-{string_to_filename(disease)}"

    if diffusion_time is not None:
        return
    if n_iters is not None:
        score_path += f"-{n_iters}_iters"
    if temp is not None:
        score_path += f"-temp_{temp}"
    if top_p is not None:
        score_path += f"-top_p_{top_p}"
    if top_k is not None:
        score_path += f"-top_k_{top_k}"
    if validation == "extended":
        score_path += f"-{validation}.csv"
    if validation == "kfold":
        score_path += f"-{K}_fold.csv"

    # Read the CSV score file as a DataFrame
    scores_df = pd.read_csv(score_path, index_col=0)

    if validation == "extended":
        # Return the array with the scores
        return scores_df.to_numpy()

    if validation == "kfold":
        # Here we need to convert the strings to float values
        scores_to_fix = scores_df.to_numpy()
        scores = np.zeros((4,4), dtype=np.float)
        scores_std = np.zeros((4,4), dtype=np.float)

        for i, size in enumerate(scores_to_fix):
            for j, value in enumerate(size):
                value = value.replace("(", "")
                value = value.replace(")", "")
                score, std = value.split(", ")
                score = float(score)
                std = float(std)

                scores[i][j] = score
                scores_std[i][j] = std

        return scores, scores_std


def average_score(algorithm=None, database=None, disease_list=None, validation=None, K=5, heat_diffusion_time=None, pdiamond_n_iters=None, pdiamond_temp=None, pdiamond_top_p=None, pdiamond_top_k=None):

    # K-Fold Validation
    if validation == "kfold":
        avg_scores = np.zeros((4,4), dtype=np.float)
        avg_scores_std = np.zeros((4,4), dtype=np.float)

        for disease in disease_list:
            try:
                scores, scores_std = get_score( algorithm=algorithm,
                                                disease=disease,
                                                database=database,
                                                validation="kfold",
                                                K=K,
                                                diffusion_time=heat_diffusion_time,
                                                n_iters=pdiamond_n_iters,
                                                temp=pdiamond_temp,
                                                top_p=pdiamond_top_p,
                                                top_k=pdiamond_top_k)

                avg_scores += (scores / len(disease_list))
                avg_scores_std += (scores_std / len(disease_list))
            except:
                continue

        # Round array values at the third decimal
        avg_scores = np.around(avg_scores, decimals=3)
        avg_scores_std = np.around(avg_scores_std, decimals=3)

        # Combine score and std
        global_scores = np.zeros((4,4), dtype=object)
        for i in range(4):
            for j in range(4):
                global_scores[i][j] = f"{avg_scores[i][j]} +/- {avg_scores_std[i][j]}"


        # Save global scores in a CSV file
        metrics = ["precision", "recall", "f1", "ndcg"]
        predicted_sizes = ["Top 25", "Top 50", "Top 100", "Top 200"]
        df = pd.DataFrame(data=global_scores, index=metrics, columns=predicted_sizes)

    # Extended Validation
    if validation == "extended":
        avg_scores = np.zeros((4,4), dtype=np.float)

        for disease in disease_list:
            try:
                scores = get_score( algorithm=algorithm,
                                    disease=disease,
                                    database=database,
                                    validation="extended",
                                    diffusion_time=heat_diffusion_time,
                                    n_iters=pdiamond_n_iters,
                                    temp=pdiamond_temp,
                                    top_p=pdiamond_top_p,
                                    top_k=pdiamond_top_k)

                avg_scores += (scores / len(disease_list))
            except:
                continue

        # Round array values at the third decimal
        avg_scores = np.around(avg_scores, decimals=3)

        # Save global scores in a CSV file
        metrics = ["precision", "recall", "f1", "ndcg"]
        predicted_sizes = ["Top 25", "Top 50", "Top 100", "Top 200"]
        df = pd.DataFrame(data=avg_scores, index=metrics, columns=predicted_sizes)

    return df

if __name__ == '__main__':

    databases = ["biogrid", "stringdb"]
    validations = ["kfold", "extended"]
    algorithms = ["diamond", "pdiamond_log", "pdiamond_entropy"]
    diseases = read_disease_file("data/disease_file.txt")
    hyperparams = {}
    temp_values = [1.0, 10.0]
    top_p_values = [0.0]
    top_k_values = [0]

    for database in databases:
        for validation in validations:
            for alg in algorithms:
                if alg == "diamond":
                    avg_df = average_score(algorithm=alg, database=database, disease_list=diseases, validation=validation, K=5)

                    outfile = f"average_results/{database}/{validation}/{alg}.csv"
                    avg_df.to_csv(outfile)


                if alg == "pdiamond_log" or alg == "pdiamond_entropy":
                    for t in temp_values:
                        for p in top_p_values:
                            for k in top_k_values:
                                avg_df = average_score( algorithm=alg,
                                                        database=database,
                                                        disease_list=diseases,
                                                        validation=validation,
                                                        K=5,
                                                        pdiamond_n_iters=10,
                                                        pdiamond_temp=t,
                                                        pdiamond_top_p=p,
                                                        pdiamond_top_k=k)

                                outfile = f"average_results/{database}/{validation}/{alg}-{10}_iters-temp_{t}-top_p_{p}-top_k_{k}.csv"
                                avg_df.to_csv(outfile)


