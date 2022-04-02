import argparse
import csv
import sys
from itertools import combinations

from matplotlib import pyplot as plt
from numpy import indices

from utils.data_utils import get_disease_genes_from_gda, string_to_filename
from utils.network_utils import *
from utils.network_utils import get_density, get_disease_LCC, get_longest_paths

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
    print('        num_iters_pdiamond   : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
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

    print('')
    print(f"============================")

    print(f"Algorithms: {algs}")
    print(f"Metrics: {metrics}")
    print(f"Precision: {p}")
    print(f"Validations: {validation_list}")
    print(f"Diseases: {disease_list}")
    print(f"Diffusion Time: {diffusion_time}")
    print(f"Num iterations pDIAMOnD: {num_iters_pdiamond}")

    print(f"============================")
    print('')

    return algs, metrics, p, validation_list, disease_list, diffusion_time, num_iters_pdiamond


# ====================== #
#   U T I L I T I E S    #
# ====================== #

def scores(alg, disease, validation="kfold", metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10):
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
        scores_path = f"results/{validation}/{alg}/{alg}_on_{string_to_filename(disease)}_{validation}_{num_iters_pdiamond}_iters.csv"
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

def who_win(alg1, alg2, disease, validation="kfold", metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10):
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

    alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)
    alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)

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

def winner_tables(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=0.005, num_iters_pdiamond=10):

    header = ["Disease", "Num disease genes", "LCC_size", "Density", "Disgenes Percentage", "Disgenes Longpath",
              "KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200",
              "EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg2 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"

    outfile = f"tables/{alg1}_vs_{alg2}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}.csv"
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
            if num_disease_genes < 1:
                disease_genes_longpath_in_LCC = get_disease_genes_longpath(disease_LCC, disease_genes)
            else:
                disease_genes_longpath_in_LCC = -1
            # print(f"{disease} longest path in LCC is: {disease_genes_longpath_in_LCC}")
            # print(f"with length: {len(disease_genes_longpath_in_LCC)}")

            # Longest path between disease genes in all the network
            # disease_genes_longpath_global = get_disease_genes_longpath(LCC_hhi, disease_genes)
            # print(f"{disease} global longest path is: {disease_genes_longpath_global}")
            # print(f"with length: {len(disease_genes_longpath_global)}")

            # ------------------------------------------------------------------------------------------


            # ** Write the data **
            data = [disease, num_disease_genes, disease_LCC_size, disease_LCC_density, disease_genes_percentage, disease_genes_longpath_in_LCC]

            for validation in validations:
                winner = who_win(alg1, alg2, disease, validation=validation, metric=metric)
                for item in winner:
                    data.append(item)

            writer.writerow(data)

    # return filename of the table
    return outfile

def how_many_time_winner(algs, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None):
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
                    if score[i] >= best_score[i]:
                        best_score[i] = score[i]
                        best_alg[i] = alg


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

    outfile = f"tables/best_algorithm_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}.csv"
    with open(outfile, "w") as f:
        writer = csv.writer(f)

        # ** Write the header **
        header = ["Algorithm",
                  "KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200",
                  "EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

        writer.writerow(header)

        for alg in algs:
            data = [alg,
                    KF_top_25.count(alg), KF_top_50.count(alg), KF_top_100.count(alg), KF_top_200.count(alg),
                    EX_top_25.count(alg), EX_top_50.count(alg), EX_top_100.count(alg), EX_top_200.count(alg)]

            writer.writerow(data)


# ============  #
#   P L O T S   #
# ============  #

def heatmap(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None):

    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"

    df = pd.read_csv(f"tables/{alg1}_vs_{alg2}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}.csv")


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
            plt.savefig(f"plots/heatmaps/{m}/{param}_{alg_pair[0]}_vs_{alg_pair[1]}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}.png")

            plt.close('all')

def absolute_heatmap(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric="f1", precision=2, diffusion_time=None, num_iters_pdiamond=None):
    alg1 = alg_pair[0]
    alg2 = alg_pair[1]

    if alg1 != "heat_diffusion" and alg2 != "heat_diffusion":
        diffusion_time = "None"
    if alg1 != "pdiamond" and alg2 != "pdiamond":
        num_iters_pdiamond = "None"

    indices = diseases

    for validation in validations:
        if validation == 'kfold':
            cols = ['KF Top 25', 'KF Top 50', 'KF Top 100', 'KF Top 200']
        if validation == 'extended':
            cols = ['EX Top 25', 'EX Top 50', 'EX Top 100', 'EX Top 200']

        compared_scores = np.zeros( (len(indices), len(cols)) )

        for idx, disease in enumerate(diseases):
            # Get algorithm scores
            alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision,
                                 diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)
            alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision,
                                 diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)

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
        figure.savefig(f'plots/heatmaps/seaborn/{validation}/{metric}/{alg1}_vs_{alg2}_{validation}_{metric}_p{precision}_diff_time_{diffusion_time}_iters_pdiamond_{num_iters_pdiamond}.png', bbox_inches='tight')

        # Close previous plots
        plt.close()



# =========== #
#   M A I N   #
# =========== #

if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs, metrics, p, validations, diseases, diffusion_time, num_iters_pdiamond = read_terminal_input(args)

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

    # for metric in metrics:
    #     for alg_pair in alg_pairs:
    #         print("                                                          ")
    #         print("----------------------------------------------------------")
    #         print(f"Comparing {alg_pair[0].upper()} and {alg_pair[1].upper()}")
    #         print("----------------------------------------------------------")
    #         # winner tables
    #         print("WINNER TABLES:")
    #         winner_table_filename = winner_tables(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)

    #         # Read algorithms score and create the heatmaps
    #         print("        ")
    #         print("HEATMAPS")
    #         absolute_heatmap(alg_pair, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)


    # How many time an algorithm is better than the other for each validation
    for metric in metrics:
        how_many_time_winner(algs, validations, diseases, hhi_df, LCC_hhi, metric=metric, precision=p, diffusion_time=diffusion_time, num_iters_pdiamond=num_iters_pdiamond)