# PROCONSUL Analysis and Comparison
In this repository you will find all the functions and the methods used to analyse the performance of PROCONSUL and compare them with the ones of DIAMOnD (article: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120, implementation: https://github.com/dinaghiassian/DIAMOnD).

The README is structured in this way: the first section describes in detail what each folder contains, the second section are the prerequisites that you need to install if you want to reproduce this code on your machine and the last is the "how to run" section in which we explain the commands to perform all the different analysis.

Please read the PROCONSUL article if you want to know in detail how PROCONSOUL works (TODO: add link) and the interpreation of the results, or, if you prefer, you can also find a simple implementation of PROCONSUL with test examples here: https://github.com/rickydeluca/PROCONSUL

## Explanation of the repository

### algorithms
It contains the implementation of PROCONSUL, DIAMOnD (take as it is from: https://github.com/dinaghiassian/DIAMOnD), and heat_diffusion (based on the following implemantation: https://github.com/idekerlab/heat-diffusion/blob/c434a01913c3a35e2c189955f135f050be246379/heat_diffusion_service.py#L39)

### data
It contains all the necessary data (databases, interactome, ppa) to test the various algorithms on different datasets.
In **biogrid** there are the Homo Sapiens interactome provided by BioGRID (BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt).
In **diamond_dataset** subfolder you can find the interactome (Interactome.tsv), the gene-disease associations (seeds.tsv) and the list of diseases (diseases.txt) used for the original DIAMOnD paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120).
In **gda** there are the gene-disease associations as provided by DisGeNET, both the curated version (curated_gene_disease_associations.tsv) used for the K-Fold Cross Validation and the complete version (all_gene_disease_associations.tsv) used for the Extended Validation.
In **stringdb** there are the full Homo Sapiens interactome as provided by StringDB (9606.protein.links.full.v11.5.txt) and the aliases file that link the StringDB protein ID with the standard ID (like the one used in BioGRID).
Finally **disease_file.txt** contains the names of the diseases on which perform the algorithm analysis, you can just edit this file to change the diseases fot the anlysis (IMPORTANT: this file won't be used if you choose to use the *diamond_dataset*, beacuse in this case it will use its own list of diseases).

### disease_analysis
It contains the results of the analysis of the disease network attributes for different datasets. They were computed in multiple settings: using only the disease genes, using the disease genes + first neighbors in the interactome and using only the first neighbors of the disease genes.

## Prerequisites

## How to run
