import argparse
import sys

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

if __name__ == "__main__":

    args = parse_args()
    alg1, alg2, validations, diseases = read_terminal_input(args)

    # for validation in validations:
    #     for disease in diseases:
    #         score1 = get_algorithm_score(algorithm=alg1, disease=disease, validation=validation)
    #         score2 = get_algorithm_score(algorithm=alg2, disease=disease, validation=validation)

    # TODO: Implement algorithm comparison