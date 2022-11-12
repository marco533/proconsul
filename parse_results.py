import sys
import argparse

import numpy as np
import pandas as pd

from utils.data_utils import string_to_filename

# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():
    print(' ')
    print('        usage: python3 parse_results.py --algs --metric --validation --disease_file --database --proconsul_n_rounds --proconsul_temp --proconsul_top_p --proconsul_top_k')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms for which parse the results.')
    print('                                   They can be: "diamond" or "proconsul" (default: all)')
    print('        metric                   : Metric to use parse the results. It can be')
    print('                                   "precision", "recall", "f1" or "ndcg".')
    print('                                   If all, parse the results for each metric. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                   If all, perform both the validations. (default: all)')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison.')
    print('                                   (default: "data/diamond_dataset/diseases.txt).')
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset".')
    print('                                   (default: "diamond_dataset)')
    print('        proconsul_n_rounds       : How many different rounds PROCONSUL will do to reduce statistical fluctuation.')
    print('                                   If you insert a list of values multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 10)')
    print('        proconsul_temp           : Temperature value for the PROCONSUL softmax function.')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 1.0)')
    print('        proconsul_top_p          : Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 0.0)')
    print('        proconsul_top_k          : Length of the pvalues subset for the PROCONSUL top-k sampling. If 0 no top-k sampling.')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 0)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "proconsul"],
                    help='List of algorithms for which compute the average results. (default: ["diamond", "proconsul"])')
    parser.add_argument('--metrics', nargs='+', default=['precision', 'recall', 'f1', 'ndcg'],
                    help='Metrics to use for which compute the average. (default: recall')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation on which test the algorithms. It can be: "kfold", "extended" or "all". If all, perform both the validations. (default: all)')
    parser.add_argument('--disease_file', type=str, default="data/diamond_dataset/diseases.txt",
                    help='Relative path to the file containing the disease names to use for the comparison. (default: "data/diamond_dataset/diseases.txt")')
    parser.add_argument('--database', type=str, default="diamond_dataset",
                    help='Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset". (default: "diamond_dataset")')
    parser.add_argument('--proconsul_n_rounds', type=int, nargs='+', default=[10],
                    help='How many different rounds PROCONSUL will do to reduce statistical fluctuation. If you insert a list of values multiple version of PROCONSUL will be run. One for each value. (default: 10)')
    parser.add_argument('--proconsul_temp', type=float, nargs='+', default=[1.0],
                    help='Temperature value for the PROCONSUL softmax function. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 1.0)')
    parser.add_argument('--proconsul_top_p', type=float, nargs='+', default=[0.0],
                    help='Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 0.0)')
    parser.add_argument('--proconsul_top_k', type=int, nargs='+', default=[0],
                    help='Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 0)')
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
    
    # 0. Read parsed values
    algs                = args.algs
    metrics             = args.metrics
    validation          = args.validation
    disease_file        = args.disease_file
    database_name       = args.database
    proconsul_n_rounds  = args.proconsul_n_rounds
    proconsul_temp      = args.proconsul_temp
    proconsul_top_p     = args.proconsul_top_p
    proconsul_top_k     = args.proconsul_top_k

    # 1. Check algorithm names
    for alg in algs:

        if alg not in ["diamond", "proconsul"]:
            print(f"ERROR: {alg} is not a valid algorithm!")
            print_usage()
            sys.exit(0)

    # 2. Check metric
    for metric in metrics:
        if metric not in ["precision", "recall", "f1", "ndcg"]:
            print(f"ERROR: {metric} is no valid metric!")
            print_usage()
            sys.exit(1)

    # 3. Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(1)

    if validation == 'all':
        validations = ['kfold', 'extended']
    else:
        validations = [validation]

    # 4. Get diesease list from file
    try:
        diseases = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        print_usage()
        sys.exit(1)

    # if empty, exit with error
    if len(diseases) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(1)

    # 5. Check database
    if database_name not in ["biogrid", "stringdb", "pnas", "diamond_dataset"]:
        print("ERROR: no valid database name")
        print_usage()
        sys.exit(1)

    if database_name == "biogrid":
        database_path = "data/biogrid/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"

    if database_name == "stringdb":
        database_path = "data/stringdb/9606.protein.links.full.v11.5.txt"

    if database_name == "pnas":
        database_path = "data/pnas/pnas.2025581118.sd02.csv"
    
    if database_name == "diamond_dataset":
        database_path = "data/diamond_dataset/Interactome.tsv"

    # 6. Check PROCONSUL number of round
    for round in proconsul_n_rounds:
        if round <= 0:
            print(f"ERROR: The number of rounds must be greater or equal 1.")
            print_usage()
            sys.exit(1)

    # 7. Check PROCONSUL temperatures
    for temp in proconsul_temp:
        if temp < 0:
            print(f"ERROR: The temperature must be greater or equal 0.")
            print_usage()
            sys.exit(1)
        
        # If temp = 0 replace it with very samll number
        # to avoid nan values
        if temp == 0:
            temp = 1e-40

    # 8. Check PROCONSUL top-p sampling
    for top_p in proconsul_top_p:
        if top_p < 0 or top_p > 1:
            print("ERROR: top_p must be in [0,1]")
            print_usage()
            sys.exit(1)

    # 9. Check PROCONSUL top-k sampling
    for top_k in proconsul_top_k:
        if top_k < 0:
            print("ERROR: top_k must be greater or equal 0.")
            print_usage()
            sys.exit(1)


    # 11. Print all the parsed inputs.
    print('                                                    ')
    print(f"===================================================")
    print(f"Algorithms: {algs}"                                 )
    print(f"Metric: {metrics}"                                   )
    print(f"Validations: {validations}"                         )
    print(f"Diseases: {len(diseases)}"                          )
    print(f"Database name: {database_name}"                     )
    print(f"Database path: {database_path}"                     )
    print(f"PROCONSUL number of rounds: {proconsul_n_rounds}"   )
    print(f"PROCONSUL temperatures: {proconsul_temp}"           )
    print(f"PROCONSUL top-p: {proconsul_top_p}"                 )
    print(f"PROCONSUL top-k: {proconsul_top_k}"                 )
    print(f"===================================================")
    print('                                                    ')

    return algs, metrics, validations, diseases, database_name, database_path, proconsul_n_rounds, proconsul_temp, proconsul_top_p, proconsul_top_k

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

def get_score(algorithm=None, disease=None, database=None, validation=None, K=None, metric=None, diffusion_time=None, n_rounds=None, temp=None, top_p=None, top_k=None):

    # Get the relative path to the algorithm score
    score_path = f"results/{database}/{validation}/{algorithm}/{algorithm}__{string_to_filename(disease)}"

    if diffusion_time is not None:
        return
    if n_rounds is not None:
        score_path += f"__{n_rounds}_rounds"
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


def average_score(algorithm=None, database=None, disease_list=None, validation=None, K=5, heat_diffusion_time=None, proconsul_n_rounds=None, proconsul_temp=None, proconsul_top_p=None, proconsul_top_k=None):

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
                                            n_rounds=proconsul_n_rounds,
                                            temp=proconsul_temp,
                                            top_p=proconsul_top_p,
                                            top_k=proconsul_top_k)

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
                                n_rounds=proconsul_n_rounds,
                                temp=proconsul_temp,
                                top_p=proconsul_top_p,
                                top_k=proconsul_top_k)

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

def average_results(database=None, validations=None, algorithms=None, diseases=None,
                    temp_values=None, top_p_values=None, top_k_values=None, n_rounds=None):
    
    for validation in validations:
        for alg in algorithms:
            if alg == "diamond":
                avg_df = average_score(algorithm=alg, database=database, disease_list=diseases, validation=validation, K=5)

                outfile = f"parsed_results/average_results/{database}/{validation}/{alg}.csv"
                avg_df.to_csv(outfile)


            if alg == "proconsul":
                for r in n_rounds:
                    for t in temp_values:
                        for p in top_p_values:
                            for k in top_k_values:
                                avg_df = average_score( algorithm=alg,
                                                        database=database,
                                                        disease_list=diseases,
                                                        validation=validation,
                                                        K=5,
                                                        proconsul_n_rounds=r,
                                                        proconsul_temp=t,
                                                        proconsul_top_p=p,
                                                        proconsul_top_k=k)

                                outfile = f"parsed_results/average_results/{database}/{validation}/{alg}__{r}_rounds__temp_{t}__top_p_{p}__top_k_{k}.csv"
                                avg_df.to_csv(outfile)


def disease_scores_table(database=None, validations=None, K=None, metrics=None, algorithms=None, diseases=None,
                            temp_values=None, top_p_values=None, top_k_values=None, n_rounds=None):

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

            
            if alg == "proconsul":
                for r in n_rounds:
                    for t in temp_values:
                        for p in top_p_values:
                            for k in top_k_values:
                                for metric in metrics:
                                
                                    outfile = f"parsed_results/disease_results/{database}/{validation}/{alg}__{r}_rounds__temp_{t}__top_p_{p}__top_k_{k}__{metric}.csv"
                                    
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
                                                                            n_rounds=r, temp=t, top_p=p, top_k=k)
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
                                                                n_rounds=r, temp=t, top_p=p, top_k=k)
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
    
    # Read input
    args = parse_args()
    algs, metrics, validations, diseases, database_name, database_path, proconsul_n_rounds, proconsul_temp, proconsul_top_p, proconsul_top_k = read_terminal_input(args)

    average_results(database=database_name, validations=validations, algorithms=algs, diseases=diseases,
                    temp_values=proconsul_temp, top_p_values=proconsul_top_p, top_k_values=proconsul_top_k, n_rounds=proconsul_n_rounds)

    disease_scores_table(database=database_name, validations=validations, algorithms=algs, diseases=diseases,
                    temp_values=proconsul_temp, top_p_values=proconsul_top_p, K=5, metrics=metrics, top_k_values=proconsul_top_k, n_rounds=proconsul_n_rounds)

    print("Done!")