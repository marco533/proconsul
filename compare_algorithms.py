import argparse
import csv
import sys
from itertools import combinations

from matplotlib import pyplot as plt
from numpy import indices

from utils.data_utils import get_disease_genes_from_gda, string_to_filename
from utils.network_utils import *
from utils.network_utils import get_density, get_disease_LCC, get_longest_path_for_a_disease_LCC, get_longest_path_for_a_disease_interactome

import seaborn as sns

# ======================= #
#   R E A D   I N P U T   #
# ======================= #

def print_usage():
    print(' ')
    print('        usage: python3 compare_algorithms.py --algs --metrics --validation --disease_file --p --diffusion_time --num_iters_pdiamond')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms to compare. They can be "diamond", "pdiamond", "heat_diffusion"')
    print('                                   (default: all')
    print('        metrics                  : Metrics used for the comparison. It can be "precision", "recall", "f1" and "ndcg"')
    print('                                   (default: all')
    print('        p                        : Decimal digit precision (default: 2)')
    print('        validation               : type of validation on which compare the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                   If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        diffusion_time           : Diffusion time for heat_diffusion algorithm.')
    print('                                   (default: 0.005)')
    print('        num_iters_pdiamond       : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print('        pdiamond_mode            : Run the classic or the alternative version of pdiamond')
    print('                                   (default: classic)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Get algorithms to compare and on which validation.')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "pdiamond", "heat_diffusion"],
                    help='List of algorithms to compare (default: all)')
    parser.add_argument('--metrics', nargs='+', default=["precision", "recall", "f1", "ndcg"],
                    help='Metrics for comparison. (default: all')
    parser.add_argument('--p', type=int, default=2,
                    help='Decimal digit precision (default: 2)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation on which compare the algorithms (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file containing the disease names to use for the comparison (default: "data/disease_file.txt)')
    parser.add_argument('--diffusion_time', type=float, default=0.005,
                    help='Diffusion time for heat_diffusion algorithm. (default: 0.005')
    parser.add_argument('--num_iters_pdiamond', type=int, default=10,
                    help='Number of iteration for pDIAMOnD. (default: 10)')
    parser.add_argument('--pdiamond_mode', type=str, default="classic",
                    help='pDIAMOnD mode (default: classic)')
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
    algs            = args.algs
    metrics         = args.metrics
    p               = args.p
    validation      = args.validation
    disease_file    = args.disease_file
    diffusion_time  = args.diffusion_time
    num_iters_pdiamond = args.num_iters_pdiamond
    pdiamond_mode       = args.pdiamond_mode

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

    # check if the algorithm names are valid
    for alg in algs:
        if alg not in ["diamond", "pdiamond", "heat_diffusion"]:
            print(f"ERROR: {alg} is not a valid algorithm!")
            print_usage()
            sys.exit(0)

    # check if the validation name is valid
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(0)

    # get the list of validations
    if validation == 'all':
        validation_list = ['kfold', 'extended']
    else:
        validation_list = [validation]

    # test diffusion time
    if diffusion_time < 0:
        print(f"ERROR: diffusion time must be greater than 0")
        sys.exit(0)

    # test num_iters_pdiamond
    if num_iters_pdiamond <= 0:
        print(f"ERROR: num_iters_pdiamond must be greater or equal of 1")
        sys.exit(0)

    # test pdiamond mode
    if pdiamond_mode not in ["classic", "alternative"]:
        print(f"ERROR: No valid mode for pdiamond, choose between 'classic' or 'alternative'")
        sys.exit(0)

    print('')
    print(f"============================")

    print(f"Algorithms: {algs}")
    print(f"Metrics: {metrics}")
    print(f"Precision: {p}")
    print(f"Validations: {validation_list}")
    print(f"Diseases: {len(disease_list)}")
    print(f"Diffusion Time: {diffusion_time}")
    print(f"Num iterations pDIAMOnD: {num_iters_pdiamond}")
    print(f"pDIAMOnD mode: {pdiamond_mode}")
    print(f"============================")
    print('')

    return algs, metrics, p, validation_list, disease_list, diffusion_time, num_iters_pdiamond, pdiamond_mode


# ====================== #
#   U T I L I T I E S    #
# ====================== #

def scores(alg, disease, validation="kfold", metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10, pdiamond_mode="classic"):
    """
    Return the scores of 'alg' on 'disease' of the
    'validation' method.
    Input:
    -------
        - alg: algorithm's name
        - disease: disease's name
        - validation: validation methon
        - metric: metric name (precision, recall, f1, ndcg)
    -------
    Return:
        - scores: A list with n elements,
                  one for each number of predicted gens.
    """

    # Read the score DataFrame
    if alg == "diamond":
        scores_path = f"results/{validation}/{alg}/{alg}_on_{string_to_filename(disease)}_{validation}.csv"

    if alg == "pdiamond":
        if pdiamond_mode == "classic":
            scores_path = f"results/{validation}/{alg}/{alg}_on_{string_to_filename(disease)}_{validation}_{num_iters_pdiamond}_iters.csv"
        if pdiamond_mode == "alternative":
            scores_path = f"results/{validation}/{alg}/{alg}_{pdiamond_mode}_on_{string_to_filename(disease)}_{validation}_{num_iters_pdiamond}_iters.csv"

    if alg == "heat_diffusion":
        scores_path = f"results/{validation}/{alg}/{alg}_on_{string_to_filename(disease)}_{validation}_diff_time_{diffusion_time}.csv"


    scores_df = pd.read_csv(scores_path, index_col=0)

    # Get values associated to the given metric in array fotm
    values = scores_df.loc[metric].values

    # Init final score of the metric
    scores = np.zeros(len(values))

    # Extract scores
    for idx, value in enumerate(values):
        # if is 'string' then we need to split
        # 'mean' and 'std_deviation'
        if type(value) == str:
            # 1. Clean the string
            value = value.replace("(", "")
            value = value.replace(")", "")

            # 2. Split 'mean' and 'std'
            mean, std = value.split(", ")

            # 3. Convert to float
            mean = float(mean)
            std = float(std)

            # 4. Round according 'precision'
            mean = round(mean, precision)
            std = round(std, precision)

            # 4. Save 'mean' into 'scores' list
            scores[idx] = mean

        else:   # no string value
            # 1. Simply save the rounded value into 'scores'
            scores[idx] = round(value, precision)

    return scores

def get_disease_param(disease, param, table=None):
    """
    Given the name of a disease and the name of a parameter,
    Return the value of the parameter of this disease.
    If we pass a table filename, read those values from that table.
    """

    if table is not None:
        if param == "Disease":
            return disease
        else:
            df = pd.read_csv(table, index_col=0)
            value = df.at[disease, param]
            if (type(value) == str):
                print(f"df[{disease}][{param}] is a string!")
                sys.exit(0)
            return df.at[disease, param]

    else:
        if param == "Disease":
            return disease
        elif param == "Num disease genes":
            return len(get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease))
        elif param == "LCC_size":
            disease_LCC = get_disease_LCC(hhi_df, disease)
            disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()
            return nx.number_of_nodes(disease_LCC)
        elif param == "Density":
            disease_LCC = get_disease_LCC(hhi_df, disease)
            disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()
            return get_density(disease_LCC)
        elif param == "Disgenes Percentage":
            disease_LCC = get_disease_LCC(hhi_df, disease)
            disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()
            disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease)
            return get_genes_percentage(disease_genes, disease_LCC)
        elif param == "Disgenes Longpath":
            disease_LCC = get_disease_LCC(hhi_df, disease)
            disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()
            return disease_LCC, disease_genes
        elif param == "Longpath LCC":
            return get_longest_path_for_a_disease_LCC(disease)
        elif param == "Longpath interactome":
            return get_longest_path_for_a_disease_interactome(disease)
        else:
            print(f"ERROR: {param} is no valid parameter.")
            sys.exit(0)

def who_win(alg1, alg2, disease, validation="kfold", metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10, pdiamond_mode="classic"):
    """
    Compare the results of 'alg1' and 'alg2' on 'disease'
    with a specific 'validation' method.
    Input:
    --------
        - alg1, alg2:
                Names of the algorithm to compare
        - disease:
                Name of the disease
        - validation:
                The valdation method to analyze
        - metric:
                Metric name (precision, recall, f1, ndcg)
    --------
    Return:
        - winner_name_and_detachment:
                A list with one element for each number of predicted genes
                and each element is composed by 2 entries:
                    * name: name of the winner algorithm
                    * victory_margin: by how much the algorithm has won.

    """

    alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)
    alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)

    winner_name_and_by_how_much = []

    # Decree the winner for each prediction size
    # (Top 25, Top 50, Top 100, Top 200)
    for i in range(len(alg1_scores)):
        if alg1_scores[i] > alg2_scores[i]:
            winner = alg1
            by_how_much = alg1_scores[i] - alg2_scores[i]
        elif alg1_scores[i] < alg2_scores[i]:
            winner = alg2
            by_how_much = alg2_scores[i] - alg1_scores[i]
        else:
            winner = "draw"
            by_how_much = 0.0

        # Save the tuple in 'winner_list'
        winner_name_and_by_how_much.append((winner, round(by_how_much, precision)))

    return winner_name_and_by_how_much

def comparison_matrix(data_array, alg_pair, mode):

    def score_advantage(str):
        str = str.replace(")", "")
        word,value = str.split(", ")
        value = float(value)

        return value

    [rows, columns] = np.shape(data_array)
    num_matrix = np.zeros((rows, columns))

    for j in range(columns):
        for i in range(rows):
            if alg_pair[1] in data_array[i][j]:
                if mode == "absolute":
                    num_matrix[i][j] = -1
                else:
                    value = score_advantage(data_array[i][j])
                    num_matrix[i][j] = value*(-1)

            elif 'draw' in data_array[i][j]:
                if mode == "absolute":
                    num_matrix[i][j] = 0
                else:
                    value = score_advantage(data_array[i][j])
                    num_matrix[i][j] = value
            else:
                if mode == "absolute":
                    num_matrix[i][j] = 1
                else:
                    value = score_advantage(data_array[i][j])
                    num_matrix[i][j] = value

    return num_matrix

# =============== #
#   T A B L E S   #
# =============== #

def winner_tables(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10, pdiamond_mode="classic"):

    header = ["Disease", "Num disease genes", "LCC_size", "Density", "Disgenes Percentage", "Longpath LCC", "Longpath interactome",
              "KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200",
              "EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"
        pdiamond_mode = "None"

    outfile = f"tables/{alg1}_vs_{alg2}/{alg1}_vs_{alg2}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.csv"
    with open(outfile, "w") as f:

        writer = csv.writer(f)

        # ** Write the header **
        writer.writerow(header)

        # Compare algorithms for each disease and for each validation
        for idx, disease in enumerate(diseases):
            print(f"Disease {idx+1}/{len(diseases)}")

            # Disease genes
            disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease)
            num_disease_genes = len(disease_genes)

            # Disease LCC
            disease_LCC = get_disease_LCC(hhi_df, disease)
            disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()

            # Disease LCC size
            disease_LCC_size = nx.number_of_nodes(disease_LCC)

            # Disease genes percentage
            disease_genes_percentage = get_genes_percentage(disease_genes, disease_LCC)
            # print(f"{disease} disease genes percentage = {disease_genes_percentage}")

            # Disease LCC density
            disease_LCC_density = get_density(disease_LCC)
            # print(f"{disease} LCC density = {disease_LCC_density}")

            # Longest path between disease genes in the LCC
            disease_genes_longpath_in_LCC = get_longest_path_for_a_disease_LCC(disease)
            # print(f"{disease} longest path in LCC is: {disease_genes_longpath_in_LCC}")
            # print(f"with length: {len(disease_genes_longpath_in_LCC)}")

            # Longest path between disease genes in all the network

            disease_genes_longpath_in_interactome = get_longest_path_for_a_disease_interactome(disease)
            # print(f"{disease} global longest path is: {disease_genes_longpath_in_interactome}")
            # print(f"with length: {len(disease_genes_longpath_in_interactome)}")

            # ------------------------------------------------------------------------------------------


            # ** Write the data **
            data = [disease, num_disease_genes, disease_LCC_size, disease_LCC_density, disease_genes_percentage, disease_genes_longpath_in_LCC, disease_genes_longpath_in_interactome]

            for validation in validations:
                winner = who_win(alg1, alg2, disease, validation=validation, metric=metric)
                for item in winner:
                    data.append(item)

            writer.writerow(data)

    # return filename of the table
    return outfile

def how_many_time_winner(algs, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None, pdiamond_mode=None):
    """
    Create a table that represents, for each algorithm, how many time
    it is the best algorithm
    """

    # Num of diseases
    N = len(diseases)

    KF_top_25   = []
    KF_top_50   = []
    KF_top_100  = []
    KF_top_200  = []

    EX_top_25   = []
    EX_top_50   = []
    EX_top_100  = []
    EX_top_200  = []

    for validation in validations:
        for disease in diseases:

            # Find the best algorithm for this disease for each prediction size
            best_alg = np.zeros(4, dtype=object)
            best_score = np.zeros(4)
            for alg in algs:
                score = scores(alg, disease, validation=validation, metric=metric, precision=precision,
                                diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)

                for i in range(len(score)):
                    if score[i] > best_score[i]:
                        best_score[i] = score[i]
                        best_alg[i] = alg
                    elif score[i] == best_score[i]:
                        best_alg[i] = "no clear winner"
                    else:
                        continue


            if validation == 'kfold':
                KF_top_25.append(best_alg[0])
                KF_top_50.append(best_alg[1])
                KF_top_100.append(best_alg[2])
                KF_top_200.append(best_alg[3])

            if validation == 'extended':
                EX_top_25.append(best_alg[0])
                EX_top_50.append(best_alg[1])
                EX_top_100.append(best_alg[2])
                EX_top_200.append(best_alg[3])


    outfile = f"tables/best_algorithm_overall/best_algorithm_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.csv"
    with open(outfile, "w") as f:
        writer = csv.writer(f)

        # ** Write the header **
        header = ["Algorithm",
                  "KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200",
                  "EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

        writer.writerow(header)
        algs_with_draw = algs.copy()
        algs_with_draw.append("no clear winner")
        # print(algs_with_draw)
        # sys.exit(0)
        for alg in algs_with_draw:
        # for alg in algs:
            data = [alg,
                    KF_top_25.count(alg), KF_top_50.count(alg), KF_top_100.count(alg), KF_top_200.count(alg),
                    EX_top_25.count(alg), EX_top_50.count(alg), EX_top_100.count(alg), EX_top_200.count(alg)]

            writer.writerow(data)

def num_won_matches(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10, pdiamond_mode="classic"):
    """
    Given a pair of algorithm,
    Return how many time one algorithm was better than the other
    for each validation output.
    """

    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"
        pdiamond_mode = "None"

    round_dict = {"KF Top 25": [], "KF Top 50": [], "KF Top 100": [], "KF Top 200": [],
                  "EX Top 25": [], "EX Top 50": [], "EX Top 100": [], "EX Top 200": []}

    for validation in validations:

        for idx, disease in enumerate(diseases):
            # Get algorithm scores
            alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision,
                                 diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)
            alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision,
                                 diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)
            # Compare scores
            compared_scores = alg1_scores - alg2_scores

            for i in range(len(compared_scores)):
                if validation == "kfold":
                    if compared_scores[i] > 0:  # alg1 > alg2
                        if i == 0:
                            round_dict["KF Top 25"].append(alg1)
                        elif i == 1:
                            round_dict["KF Top 50"].append(alg1)
                        elif i == 2:
                            round_dict["KF Top 100"].append(alg1)
                        else:
                            round_dict["KF Top 200"].append(alg1)
                    elif compared_scores[i] < 0:  # alg1 < alg2
                        if i == 0:
                            round_dict["KF Top 25"].append(alg2)
                        elif i == 1:
                            round_dict["KF Top 50"].append(alg2)
                        elif i == 2:
                            round_dict["KF Top 100"].append(alg2)
                        else:
                            round_dict["KF Top 200"].append(alg2)
                    else:   # alg1 == alg2
                        if i == 0:
                            round_dict["KF Top 25"].append("draw")
                        elif i == 1:
                            round_dict["KF Top 50"].append("draw")
                        elif i == 2:
                            round_dict["KF Top 100"].append("draw")
                        else:
                            round_dict["KF Top 200"].append("draw")

                if validation == "extended":
                    if compared_scores[i] > 0:  # alg1 > alg2
                        if i == 0:
                            round_dict["EX Top 25"].append(alg1)
                        elif i == 1:
                            round_dict["EX Top 50"].append(alg1)
                        elif i == 2:
                            round_dict["EX Top 100"].append(alg1)
                        else:
                            round_dict["EX Top 200"].append(alg1)
                    elif compared_scores[i] < 0:  # alg1 < alg2
                        if i == 0:
                            round_dict["EX Top 25"].append(alg2)
                        elif i == 1:
                            round_dict["EX Top 50"].append(alg2)
                        elif i == 2:
                            round_dict["EX Top 100"].append(alg2)
                        else:
                            round_dict["EX Top 200"].append(alg2)
                    else:   # alg1 == alg2
                        if i == 0:
                            round_dict["EX Top 25"].append("draw")
                        elif i == 1:
                            round_dict["EX Top 50"].append("draw")
                        elif i == 2:
                            round_dict["EX Top 100"].append("draw")
                        else:
                            round_dict["EX Top 200"].append("draw")

    # Count how many time one algorithm appears in each dict key
    KF_top_25 = round_dict["KF Top 25"]
    KF_top_50 = round_dict["KF Top 50"]
    KF_top_100 = round_dict["KF Top 100"]
    KF_top_200 = round_dict["KF Top 200"]

    EX_top_25 = round_dict["EX Top 25"]
    EX_top_50 = round_dict["EX Top 50"]
    EX_top_100 = round_dict["EX Top 100"]
    EX_top_200 = round_dict["EX Top 200"]

    outfile = f"tables/{alg1}_vs_{alg2}/num_times_winner_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.csv"
    with open(outfile, "w") as f:
        writer = csv.writer(f)

        # ** Write the header **
        header = ["Algorithm",
                  "KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200",
                  "EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

        writer.writerow(header)
        algs_with_draw =list(alg_pair).copy()
        algs_with_draw.append("draw")
        # print(algs_with_draw)
        # sys.exit(0)
        for alg in algs_with_draw:
        # for alg in algs:
            data = [alg,
                    KF_top_25.count(alg), KF_top_50.count(alg), KF_top_100.count(alg), KF_top_200.count(alg),
                    EX_top_25.count(alg), EX_top_50.count(alg), EX_top_100.count(alg), EX_top_200.count(alg)]

            writer.writerow(data)



# ============  #
#   P L O T S   #
# ============  #

def heatmap(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None, pdiamond_mode="classic"):

    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"
        pdiamond_mode = "None"

    df = pd.read_csv(f"tables/{alg1}_vs_{alg2}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.csv")


    parameters = ["LCC_size", "Density", "Disgenes Percentage"]

    #This counter indicates the column where the first parameter is located
    column_counter = 2
    mode = ["absolute", "gradient"]

    for m in mode:

        for param in parameters:

            sorted_df = df.sort_values(by = param)
            data = sorted_df.values.tolist()

            #We convert our data to a convenient format
            data_array = np.array(data)
            yaxis = data_array[0,:]
            yaxis = np.delete(yaxis, [0,1,2,3,4,5])
            yaxis = ['KF Top 25', 'KF Top 50', 'KF Top 100', 'KF Top 200', 'EX Top 25', 'EX Top 50', 'EX Top 100', 'EX Top 200']
            xaxis = data_array[:,column_counter]
            xaxis = np.delete(xaxis, 0)
            data_array = np.delete(data_array,[0,1,2,3,4,5], axis=1)
            data_array = np.delete(data_array,0, axis=0)

            column_counter +=  1

            #This function generates a numerical matrix that is used for the heat map generation

            num_matrix = comparison_matrix(data_array, alg_pair, m)

            #Tranpose the matrix so the plot fits better on the screen
            num_matrix = np.transpose(num_matrix)

            #We plot the results
            cmap = plt.get_cmap('bwr')
            fig, ax = plt.subplots()
            img = ax.imshow(num_matrix, cmap = cmap)

            ax.set_xticks(np.arange(len(xaxis)))
            ax.set_yticks(np.arange(len(yaxis)))
            ax.set_yticklabels(yaxis)
            ax.set_xticklabels(xaxis)

            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
            plt.colorbar(img, shrink=0.5)
            plt.text(97,20, f"{alg_pair[1]} > {alg_pair[0]}")
            plt.text(96,-12, f"{alg_pair[0]} > {alg_pair[1]}")

            fig.suptitle(f"{param}: {alg_pair[0]} vs {alg_pair[1]}")

            # for i in range(len(yaxis)):
            #     for j in range(len(xaxis)):
            #         text = ax.text(j, i, num_matrix[i, j], ha="center", va="center", color="w")

            #plt.show()
            plt.savefig(f"plots/heatmaps/{m}/{param}_{alg_pair[0]}_vs_{alg_pair[1]}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.png")

            plt.close('all')

def absolute_heatmap(alg_pair, validations, diseases, table=None, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None, pdiamond_mode="classic", cluster=False):
    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"
        pdiamond_mode = "None"

    if cluster:
        # Define disease parameters
        params = ["Disease", "LCC_size", "Num disease genes", "Density", "Disgenes Percentage", "Longpath LCC", "Longpath interactome"]
        for param in params:
            # Create a dictionary disease: param
            disease_param_dict = {}
            for disease in diseases:
                disease_param_dict[disease] = get_disease_param(disease, param, table=table)

            indices = disease_param_dict.values()

            for validation in validations:
                if validation == 'kfold':
                    cols = ['KF Top 25', 'KF Top 50', 'KF Top 100', 'KF Top 200']
                if validation == 'extended':
                    cols = ['EX Top 25', 'EX Top 50', 'EX Top 100', 'EX Top 200']

                compared_scores = np.zeros( (len(indices), len(cols)) )

                for idx, disease in enumerate(diseases):
                    # Get algorithm scores
                    alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision,
                                        diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)
                    alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision,
                                        diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)

                    # Compare scores
                    compared_scores[idx] = alg1_scores - alg2_scores

                # Create DataFrame with the compared scores
                compared_scores_df = pd.DataFrame(compared_scores, index=indices, columns=cols)

                # Plot ClusterMap
                cmap = sns.diverging_palette(220, 20, as_cmap=True)
                g = sns.clustermap(compared_scores_df, cmap=cmap, center=0)

                g.savefig(f'plots/clustered_heatmaps/{alg1}_vs_{alg2}/{alg1}_vs_{alg2}__{validation}__{string_to_filename(param)}__{metric}__p{precision}__diff_time_{diffusion_time}__iters_pdiamond_{num_iters_pdiamond}__pdiamond_mode_{pdiamond_mode}.png', bbox_inches='tight')

                # Close previous plots
                plt.close()


    else:
        # Define disease parameters
        params = ["Disease", "Num disease genes", "LCC_size", "Density", "Disgenes Percentage", "Longpath LCC", "Longpath interactome"]
        # params = ["Longpath LCC", "Longpath interactome"]
        for param in params:
            # Create a dictionary disease: param
            disease_param_dict = {}
            for disease in diseases:
                disease_param_dict[disease] = get_disease_param(disease, param, table=table)

            # Sort the dictionary keys according to their values
            sorted_tuples = sorted(disease_param_dict.items(), key=lambda item: item[1])
            sorted_dict = {k: v for k, v in sorted_tuples}

            indices = sorted_dict.values()
            sorted_diseases = sorted_dict.keys()

            for validation in validations:
                if validation == 'kfold':
                    cols = ['KF Top 25', 'KF Top 50', 'KF Top 100', 'KF Top 200']
                if validation == 'extended':
                    cols = ['EX Top 25', 'EX Top 50', 'EX Top 100', 'EX Top 200']

                compared_scores = np.zeros( (len(indices), len(cols)) )

                for idx, disease in enumerate(sorted_diseases):
                    # Get algorithm scores
                    alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision,
                                        diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)
                    alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision,
                                        diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)

                    # Compare scores
                    # compared_scores[idx] = alg1_scores  # Copy the alg1_scores
                    # comp_bool_array = alg1_scores < alg2_scores # True when alg1 < alg2, False otherwise
                    # compared_scores[idx][comp_bool_array] = -alg2_scores[comp_bool_array] # When alg1 < alg2 substitute the final score with -alg2

                    compared_scores[idx] = alg1_scores - alg2_scores

                # Create DataFrame with the compared scores
                compared_scores_df = pd.DataFrame(compared_scores, index=indices, columns=cols)

                # Plot HeatMap
                cmap = sns.diverging_palette(220, 20, as_cmap=True)
                hm = sns.heatmap(compared_scores_df, annot=False, cmap=cmap, center=0, vmin=-0.1, vmax=0.1)

                figure = hm.get_figure()
                figure.savefig(f'plots/heatmaps/absolute_score/{validation}/{metric}/{alg1}_vs_{alg2}_{string_to_filename(param)}_{validation}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.png', bbox_inches='tight')

                # Close previous plots
                plt.close()


def clustered_heatmap(alg_pair, validations, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None, pdiamond_mode="classic"):
    """
    Given a table with the scores for two algorithms
    and the properties for each disease,
    build an heatmap that show the winner for
    the selected <validation>, the selected
    <output size> and the slected <metric>,
    clusterized wrt the disease properties.
    """

    # Extract the two algorithms in the pair
    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    # Set value for diffusion_time and num_iters_pdiamond
    # according the algorithms into the pair
    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"
        pdiamond_mode = "None"

    for validation in validations:
        # List the possible prediction sizes
        # according to <validation>
        if validation == "kfold":
            predictions = ["KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200"]
        elif validation == "extended":
            predictions = ["EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]
        else:
            print("ERROR: No valid validation name")
            sys.exit(0)

        # Import the dataset according the specified parameters
        dataset_filename = f"tables/{alg1}_vs_{alg2}/{alg1}_vs_{alg2}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.csv"
        data = pd.read_csv(dataset_filename)

        for prediction in predictions:
            # Select from data only the disease properties
            # and the winner of just one <prediction>
            clear_data = data[["Num disease genes",
                                "LCC_size",
                                "Density",
                                "Disgenes Percentage",
                                "Longpath LCC",
                                "Longpath interactome",
                                prediction]]


            # From the cleared data get only the prediction column
            # that contain the tuple (winner, by_how_much)
            winners_with_score = clear_data.pop(prediction) # get the <prediction> column and remove it from the df

            # Extract only the winner names
            winners = []
            for winner_tuple in winners_with_score:
                # print("winner_tuple: ", winner_tuple)
                winner_tuple = winner_tuple.replace("(", "")
                winner_tuple = winner_tuple.replace(")", "")
                winner, score = winner_tuple.split(", ")
                winner = winner.replace("'", "")
                score = float(score)
                # print("winner: ", winner)
                # print("score: ", score)

                # Append winner name to the winner list
                winners.append(winner)

            # Replace the popped column with this new one
            # containing only the winner names
            clear_data[prediction] = winners
            # print(clear_data.head())

            # Pop out new winners col to use it
            # as observation for heatmap
            winners = clear_data.pop(prediction)

            # We can finally build the clustered heatmap
            lut = dict(zip(["diamond", "pdiamond", "draw"], "rbg"))
            row_colors = winners.map(lut)
            print(lut)
            g = sns.clustermap(clear_data, row_colors=row_colors, standard_scale=1)

            g.savefig(f'plots/clustered_heatmaps/{alg1}_vs_{alg2}/{alg1}_vs_{alg2}__{string_to_filename(prediction)}__{metric}__p{precision}__diff_time_{diffusion_time}__iters_pdiamond_{num_iters_pdiamond}_pdiamond_mode_{pdiamond_mode}.png', bbox_inches='tight')

            # Close previous plots
            plt.close()


# =========== #
#   M A I N   #
# =========== #

if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs, metrics, p, validations, diseases, diffusion_time, num_iters_pdiamond, pdiamond_mode = read_terminal_input(args)

    # Human-Human Interactome
    biogrid = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df  = select_hhi_only(biogrid)
    hhi     = nx.from_pandas_edgelist(hhi_df,
                                      source = "Official Symbol Interactor A",
                                      target = "Official Symbol Interactor B",
                                      create_using=nx.Graph())  #nx.Graph doesn't allow duplicated edges
    # Remove self loops
    self_loop_edges = list(nx.selfloop_edges(hhi))
    hhi.remove_edges_from(self_loop_edges)

    # Largest Connected Component
    LCC_hhi = isolate_LCC(hhi)
    LCC_hhi = hhi.subgraph(LCC_hhi).copy()

    # Create all possible pairs for algorithms in "algs"
    alg_pairs = list(combinations(algs, 2))

    # For each algorithm pair
    for metric in metrics:

        print( "**************************")
        print(f"  {metric.upper()} SCORE  ")
        print( "**************************")

        for alg_pair in alg_pairs:
            print("                                                          ")
            print("----------------------------------------------------------")
            print(f"Comparing {alg_pair[0].upper()} and {alg_pair[1].upper()}")
            print("----------------------------------------------------------")

            # Winner tables
            print("WINNER TABLES:")
            winner_table_filename = winner_tables(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)

            # Heatmaps
            print("        ")
            print("HEATMAPS")
            table_filename = "tables/diamond_vs_pdiamond/diamond_vs_pdiamond_f1_p2_diff_time_None_iters_pdiamond_10_pdiamond_mode_classic.csv"
            absolute_heatmap(alg_pair, validations, diseases, table=table_filename, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode, cluster=False)

            # Clustered heatmaps
            clustered_heatmap(alg_pair, validations, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)

            # Num won matches
            num_won_matches(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)


        # How many time an algorithm is better than the other for each validation
        how_many_time_winner(algs, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond, pdiamond_mode=pdiamond_mode)