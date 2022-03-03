import sys

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import *
from get_disease_LCC import get_disease_LCC


def plot_scores_by_lcc_size(interactome, disease_list, algorithm_list, validation='kfold', metric='f1', output_size='Top 100', scatter=True, max_lcc=-1):
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
                                      from_curated=True)
                                    #   from_curated=True if validation=='kfold' else False)

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
            if max_lcc != -1 and diseases_by_lcc_size[disease] > max_lcc:
                continue
            else:
                LCC_sizes.append(diseases_by_lcc_size[disease])
                scores.append(score)

        # Add the algorithm score to the plot
        if scatter:
            plt.scatter(LCC_sizes, scores, label=algorithm)
        else:
            plt.plot(LCC_sizes, scores, label=algorithm)

    # plot legend and save as png
    plt.legend()
    plt.title(f"Scores by LCC size - Metric: {metric.upper()} | Predicted genes: {output_size} | Validation: {validation.upper()}")
    if max_lcc != -1:
        plt.savefig(f"plots/scores_by_lcc_size/{string_to_filename(output_size)}_{metric}_{validation}_max_lcc_{max_lcc}.png", bbox_inches="tight")
    else:
        plt.savefig(f"plots/scores_by_lcc_size/{string_to_filename(output_size)}_{metric}_{validation}.png", bbox_inches="tight")
    plt.close('all')
