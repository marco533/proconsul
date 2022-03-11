import argparse
import csv
import sys

from utils.data_utils import get_disease_genes_from_gda, string_to_filename
from utils.network_utils import *
from utils.network_utils import get_density, get_disease_LCC, get_longest_paths

# ======================= #
#   R E A D   I N P U T   #
# ======================= #

def print_usage():
    print(' ')
    print('        usage: python3 compare_algorithms.py --alg1 --alg2 --validation --disease_file')
    print('        -----------------------------------------------------------------')
    print('        alg1             : First algorithm to compare. It can be "diamond", "prob_diamond" or "heat_diffusion".')
    print('                           (default: diamond')
    print('        alg2             : Second algorithm to compare.')
    print('                           (default: heat_diffusion')
    print('        validation       : type of validation on which compare the algorithms. It can be')
    print('                           "kfold", "extended" or "all".')
    print('                           If all, perform both the validations. (default: all')
    print('        disease_file     : Relative path to the file containing the disease names to use for the comparison')
    print('                           (default: "data/disease_file.txt).')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Get algorithms to compare and on which validation.')
    parser.add_argument('--alg1', type=str, default='diamond',
                    help='First algorithm. (default: diamond)')
    parser.add_argument('--alg2', type=str, default='heat_diffusion',
                    help='Second algorithm. (default: heat_diffusion')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation on which compare the algorithms (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file containing the disease names to use for the comparison (default: "data/disease_file.txt)')
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
    alg1      = args.alg1
    alg2      = args.alg2
    validation      = args.validation
    disease_file    = args.disease_file

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
    if alg1 not in ["diamond", "prob_diamond", "heat_diffusion"]:
        print(f"ERROR: {alg1} is not a valid algorithm!")
        print_usage()
        sys.exit(0)

    if alg2 not in ["diamond", "prob_diamond", "heat_diffusion"]:
        print(f"ERROR: {alg2} is not a valid algorithm!")
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

    print('')
    print(f"============================")

    print(f"Algorithm 1: {alg1}")
    print(f"Algorithm 2: {alg2}")
    print(f"Validations: {validation_list}")
    print(f"Diseases: {disease_list}")


    print(f"============================")
    print('')

    return alg1, alg2, validation_list, disease_list


# ====================== #
#   U T I L I T I E S    #
# ====================== #

def scores(alg, disease, validation="kfold", metric="f1", precision=2):
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
    scores_path = f"results/{validation}/{alg}/{alg}_on_{string_to_filename(disease)}_{validation}.csv"
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

def who_win(alg1, alg2, disease, validation="kfold", metric="f1", precision=2):
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

    alg1_scores = scores(alg1, disease, validation=validation, metric=metric, precision=precision)
    alg2_scores = scores(alg2, disease, validation=validation, metric=metric, precision=precision)

    winner_name_and_detachment = []

    # Decree the winner for each prediction size
    # (Top 50, Top 100, Top 200, Top N)
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
        winner_name_and_detachment.append((winner, round(by_how_much, precision)))

    return winner_name_and_detachment

# =============== #
#   T A B L E S   #
# =============== #

def winner_tables(alg1, alg2, validations, diseases, metric="f1", precision=2):

    header = ["Disease", "Num disease genes", "LCC_size", "Density", "Disgenes Percentage", "Disgenes Longpath",
                "KF Top 50", "KF Top 100", "KF Top 200", "KF Top N",
                "EX Top 50", "EX Top 100", "EX Top 200", "EX Top N"]

    with open(f"tables/{alg1}_vs_{alg2}_{metric}_p{precision}.csv", "w") as f:

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


# =========== #
#   M A I N   #
# =========== #

if __name__ == "__main__":

    # Read input
    args = parse_args()
    alg1, alg2, validations, diseases = read_terminal_input(args)

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

    for metric in ["f1"]:
        winner_tables(alg1, alg2, validations, diseases, metric=metric, precision=2)


