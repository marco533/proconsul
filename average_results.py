import sys

import numpy as np
import pandas as pd
import networkx as nx

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

def get_score(algorithm=None, disease=None, validation=None, K=None, diffusion_time=None, n_iters=None, temp=None, top_p=None, top_k=None):

    # Get the relative path to the algorithm score
    score_path = f"results/{validation}/{algorithm}/{algorithm}-{string_to_filename(disease)}"

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

if __name__ == '__main__':

    diseases = read_disease_file("data/disease_file.txt")
    temp_values = [1.0, 10.0, 100.0]
    top_p_values = [0.0]
    top_k_values = [0]

    # 5-Fold Cross Validation
    print("5-Fold\n")

    # DIAMOnD
    print("DIAMOnD\n")

    avg_scores = np.zeros((4,4), dtype=np.float)
    avg_scores_std = np.zeros((4,4), dtype=np.float)

    for disease in diseases:
        try:
            scores, scores_std = get_score( algorithm="diamond",
                                            disease=disease,
                                            validation="kfold",
                                            K=5)

            avg_scores += (scores / len(diseases))
            avg_scores_std += (scores_std / len(diseases))
        except:
            continue

    print(avg_scores)
    print(avg_scores_std)
    print("")

    # pDIAMOnD 2
    print("pDIAMOnD 2\n")

    for T in temp_values:
        for p in top_p_values:
            for k in top_k_values:

                avg_scores = np.zeros((4,4), dtype=np.float)
                avg_scores_std = np.zeros((4,4), dtype=np.float)

                for disease in diseases:
                    try:
                        scores, scores_std = get_score( algorithm="pdiamond_log",
                                                        disease=disease,
                                                        validation="kfold",
                                                        K=5,
                                                        n_iters=10,
                                                        temp=T,
                                                        top_p=p,
                                                        top_k=k)

                        avg_scores += (scores / len(diseases))
                        avg_scores_std += (scores_std / len(diseases))
                    except:
                        continue

                print("Temp: ", T)
                print(avg_scores)
                print(avg_scores_std)
                print("")






    # Extended
    print("Extended\n")

    # DIAMOnD
    print("DIAMOnD\n")

    avg_scores = np.zeros((4,4), dtype=np.float)

    for disease in diseases:
        try:
            scores = get_score( algorithm="diamond",
                                disease=disease,
                                validation="extended")

            avg_scores += (scores / len(diseases))

        except:
            continue

    print(avg_scores)
    print("")

    # pDIAMOnD 2
    print("pDIAMOnD 2\n")

    for T in temp_values:
        for p in top_p_values:
            for k in top_k_values:

                avg_scores = np.zeros((4,4), dtype=np.float)

                for disease in diseases:
                    try:
                        scores = get_score( algorithm="pdiamond_log",
                                            disease=disease,
                                            validation="extended",
                                            n_iters=10,
                                            temp=T,
                                            top_p=p,
                                            top_k=k)

                        avg_scores += (scores / len(diseases))

                    except:
                        continue

                print("Temp: ", T)
                print(avg_scores)
                print("")
