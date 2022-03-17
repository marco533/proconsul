import math

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_max_dcg(len_test_genes, top):
    ''' Compute max dcg '''
    max_dcg = 0
    #it computes minimum number between the test genes lenght and how many are the considered top rankings
    #because this minimum is the number of adding to do
    iterations = min(len_test_genes, top)
    #add 1/log_result iterations times in according whit formula
    for i in range(iterations):
        argument = i + 1 + 1 #formula says ranking_pos + 1 and ranking_pos is i+1
        log_result = math.log(argument, 2)
        max_dcg += 1/log_result
    return max_dcg

def compute_metrics(all_genes, test_genes, predicted_genes, is_ranking=True):
    '''
    Given the network, the predicted genes and the test set
    compute the following metrics:
        - Precision
        - Recall
        - F1 Score
        - NDCG
    '''

    # create the arrays of true and predicted values
    num_tot_genes = len(all_genes)
    y_true = np.zeros(num_tot_genes)
    y_pred = np.zeros(num_tot_genes)

    for idx, gene in enumerate(all_genes):
        if gene in test_genes: # true positive + false negative (ground truth)
            y_true[idx] = 1
        if gene in predicted_genes:
            y_pred[idx] = 1

    # compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ndcg = 0.0

    if is_ranking:
        ranking_position = 0            # position in the ranking

        for gene in predicted_genes:
            ranking_position +=1

            # for each gene in the predicted ranking, check if it was in the test set
            if gene in test_genes:
                argument = ranking_position + 1
                log_result = math.log(argument, 2)
                ndcg += 1/log_result


        # Normalization of DCG
        #
        # compute max_dcg
        max_dcg = compute_max_dcg(len(test_genes), len(predicted_genes))


        # and normalize it
        try:
            ndcg = ndcg / max_dcg
        except:
            ndcg = 0

    return [precision, recall, f1, ndcg]
