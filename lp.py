from utils.network_utils import *
import numpy as np
import bct

if __name__ == "__main__":

    hhi_lcc = 'data/HHI_LCC.txt'

    genes_dict = {}

    #create a dictionary with key=gene_name and value=a unique progressive integer to identify the gene
    with open(hhi_lcc, 'r') as of:

            i=0
            for line in of:
                node1=line.strip().split(',')[0]
                node2=line.strip().split(',')[1]

                if (node1 not in genes_dict):
                    genes_dict[node1]= i
                    i+=1
                if (node2 not in genes_dict):
                    genes_dict[node2]= i
                    i+=1

    #print(len(genes_dict))

    adjacency_matrix = np.zeros((len(genes_dict), len(genes_dict)))
    #print(distance_matrix)


    #fill adjacency matrix with 1
    with open(hhi_lcc, 'r') as of:

        for line in of:
            node1=line.strip().split(',')[0]
            node2=line.strip().split(',')[1]
            gene1_id = genes_dict[node1]
            gene2_id = genes_dict[node2]

            if (gene1_id != gene2_id):
                adjacency_matrix[gene1_id,gene2_id]=1

    print(adjacency_matrix)

    # Save adjacency metric as binary file
    with open("data/adjacency_matrix", "wb") as f:
        np.save(f, adjacency_matrix)

    #compute distance matrix
    distance_matrix = bct.distance_bin(adjacency_matrix)
    print(distance_matrix)

    # Save adjacency metric as binary file
    with open("data/distance_matrix", "wb") as f:
        np.save(f, distance_matrix)





