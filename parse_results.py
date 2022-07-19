import sys

import numpy as np
import pandas as pd

from utils.data_utils import string_to_filename

# =========================
#     U T I L I T I E S
# =========================

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

def get_score(algorithm=None, disease=None, database=None, validation=None, K=None, metric=None, diffusion_time=None, n_iters=None, temp=None, top_p=None, top_k=None):

    # Get the relative path to the algorithm score
    score_path = f"results/{database}/{validation}/{algorithm}/{algorithm}__{string_to_filename(disease)}"

    if diffusion_time is not None:
        return
    if n_iters is not None:
        score_path += f"__{n_iters}_iters"
    if temp is not None:
        score_path += f"__temp_{temp}"
    if top_p is not None:
        score_path += f"__top_p_{top_p}"
    if top_k is not None:
        score_path += f"__top_k_{top_k}"
    if validation == "extended":
        score_path += f"__{validation}.csv"
    if validation == "kfold":
        score_path += f"__{K}_fold.csv"

    # Read the CSV score file as a DataFrame
    scores_df = pd.read_csv(score_path, index_col=0)

    # If metric is not None, select only the results relative to that metric
    if metric:
        indices = scores_df.index.to_list()
        indices_to_remove = []
        for idx in indices:
            if idx != metric:
                indices_to_remove.append(idx)
        
        # Drop out the other metrics
        scores_df = scores_df.drop(indices_to_remove)

    if validation == "extended":
        # Return the array with the scores
        return scores_df.to_numpy()

    if validation == "kfold":
        # Here we need to convert the strings to float values
        scores_to_fix = scores_df.to_numpy()

        scores = np.zeros((scores_to_fix.shape[0], scores_to_fix.shape[1]), dtype=np.float)
        scores_std = np.zeros((scores_to_fix.shape[0], scores_to_fix.shape[1]), dtype=np.float)

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

        # Round array values at the third decimal
        avg_scores = np.around(avg_scores, decimals=3)

        # Save global scores in a CSV file
        metrics = ["precision", "recall", "f1", "ndcg"]
        predicted_sizes = ["Top 25", "Top 50", "Top 100", "Top 200"]
        df = pd.DataFrame(data=avg_scores, index=metrics, columns=predicted_sizes)

    return df




# ===============
#     A P P S
# ===============

def average_results(databases=None, validations=None, algorithms=None, diseases=None,
                    temp_values=None, top_p_values=None, top_k_values=None):
    
    for database in databases:
        for validation in validations:
            for alg in algorithms:
                if alg == "diamond":
                    avg_df = average_score(algorithm=alg, database=database, disease_list=diseases, validation=validation, K=5)

                    outfile = f"parsed_results/average_results/{database}/{validation}/{alg}.csv"
                    avg_df.to_csv(outfile)


                if alg == "pdiamond_log":
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

                                outfile = f"parsed_results/average_results/{database}/{validation}/{alg}__{10}_iters__temp_{t}__top_p_{p}__top_k_{k}.csv"
                                avg_df.to_csv(outfile)


def disease_scores_table(databases=None, validations=None, K=None, metrics=None, algorithms=None, diseases=None,
                            temp_values=None, top_p_values=None, top_k_values=None):

    for database in databases:
        for validation in validations:
            for alg in algorithms:
                
                if alg == "diamond":
                    for metric in metrics:
                        
                        outfile = f"parsed_results/disease_results/{database}/{validation}/{alg}__{metric}.csv"
                        
                        index = []
                        columns = ["Top 25", "Top 50", "Top 100", "Top 200"]
                        data = np.zeros((len(diseases), 4), dtype=object)

                        for idx, disease in enumerate(diseases):
                            index.append(disease)

                            if validation == "kfold":
                                try:
                                    score, std = get_score(algorithm=alg, disease=disease, database=database,
                                                            validation=validation, K=K, metric=metric)
                                except:
                                    score = np.ones((1,4)) * -1
                                    std = np.ones((1,4)) * -1
                                
                                # Round array values at the third decimal
                                score = np.around(score, decimals=3)
                                std = np.around(std, decimals=3)

                                # Combine score and std
                                global_score = np.zeros((1,4), dtype=object)
                                for j in range(4):
                                    global_score[0][j] = f"{score[0][j]} +/- {std[0][j]}"
                                
                                # Save into data
                                data[idx] = global_score
                            
                            if validation == "extended":
                                try:
                                    score = get_score(algorithm=alg, disease=disease, database=database,
                                                    validation=validation, metric=metric)
                                except:
                                    score = np.ones((1,4)) * -1
                                
                                # Round array values at the third decimal
                                score = np.around(score, decimals=3)

                                # Save into data
                                data[idx] = score
                        
                        # Cast DataFrame to CSV
                        df = pd.DataFrame(data=data, index=index, columns=columns)
                        df.to_csv(outfile)

                
                if alg == "pdiamond_log":
                    for t in temp_values:
                        for p in top_p_values:
                            for k in top_k_values:
                                for metric in metrics:
                                
                                    outfile = f"parsed_results/disease_results/{database}/{validation}/{alg}__{10}_iters__temp_{t}__top_p_{p}__top_k_{k}__{metric}.csv"
                                    
                                    index = []
                                    columns = ["Top 25", "Top 50", "Top 100", "Top 200"]
                                    data = np.zeros((len(diseases), 4), dtype=object)

                                    for idx, disease in enumerate(diseases):
                                        index.append(disease)

                                        # Get score
                                        if validation == "kfold":
                                            try:
                                                score, std = get_score(algorithm=alg, disease=disease, database=database,
                                                                            validation=validation, K=K, metric=metric,
                                                                            n_iters=10, temp=t, top_p=p, top_k=k)
                                            except:
                                                score = np.zeros((1,4))
                                                std = np.zeros((1,4))

                                            # Round array values at the third decimal
                                            score = np.around(score, decimals=3)
                                            std = np.around(std, decimals=3)

                                            # Combine score and std
                                            global_score = np.zeros((1,4), dtype=object)
                                            for j in range(4):
                                                global_score[0][j] = f"{score[0][j]} +/- {std[0][j]}"
                                            
                                            # Save into data
                                            data[idx] = global_score
                                        
                                        if validation == "extended":
                                            try:
                                                score = get_score(algorithm=alg, disease=disease, database=database,
                                                                validation=validation, metric=metric,
                                                                n_iters=10, temp=t, top_p=p, top_k=k)
                                            except:
                                                score = np.zeros((1,4))
                                            
                                            # Round array values at the third decimal
                                            score = np.around(score, decimals=3)
                                            
                                            # Save into data
                                            data[idx] = score
                                    
                                    # Cast DataFrame to CSV
                                    df = pd.DataFrame(data=data, index=index, columns=columns)
                                    df.to_csv(outfile)



# ===============
#     M A I N    
# ===============

if __name__ == '__main__':

    databases = ["biogrid"]
    validations = ["kfold"]
    algorithms = ["pdiamond_log", "diamond"]
    metrics = ["precision", "recall", "f1", "ndcg"]
    diseases = read_disease_file("data/disease_file_all.txt")
    # diseases = read_disease_file("data/diamond_dataset/diseases.txt")

    hyperparams = {}
    temp_values = [1.0]
    top_p_values = [0.0]
    top_k_values = [0]

    average_results(databases=databases, validations=validations, algorithms=algorithms, diseases=diseases,
                    temp_values=temp_values, top_p_values=top_p_values, top_k_values=top_k_values)

    disease_scores_table(databases=databases, validations=validations, algorithms=algorithms, diseases=diseases,
                    temp_values=temp_values, top_p_values=top_p_values, K=5, metrics=metrics, top_k_values=top_k_values)

    print("Done!")