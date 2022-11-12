# PROCONSUL Analysis and Comparison
In this repository you will find all the functions and the methods used to analyse the performance of PROCONSUL and compare them with the ones of DIAMOnD (article: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120, implementation: https://github.com/dinaghiassian/DIAMOnD).

The README is structured in this way: the first section describes in detail what each folder contains, the second section are the prerequisites that you need to install if you want to reproduce this code on your machine and the last is the "how to run" section in which we explain the commands to perform all the different analysis.

Please read the PROCONSUL article if you want to know in detail how PROCONSOUL works (TODO: add link) and the interpreation of the results, or, if you prefer, you can also find a simple implementation of PROCONSUL with test examples here: https://github.com/rickydeluca/PROCONSUL

## Explanation of the repository
In the main folder you can find:
* __analyse_disease_network.py__ performs the analysis of the disease network attributes and returns the tables (that will be store in _disease_analysis_ subfolder)
* __compare_algs.py__ produce the hetmaps of the comparison between the algorithms (stored in _plots_)
* __parse_results.py__ perfroms the average of the results over all the diseases collected by the main algorithm and the produces the table in which are stored the results for each disease (stored in _parsed_results_ subfolder).
* __plot_iteration_score.py__ produces the plots that compare how the performance of the algorithms evolve as we increase the numeber of the iterations.
* __main.py__ runs the algorithms, collect their results and run the modules to perform the validations.
* __requirements.txt__ contains the list of prerequisites needed to reproduce this analysis.

Now we will see the contents of each subfolder.

### algorithms
It contains the implementation of PROCONSUL, DIAMOnD (take as it is from: https://github.com/dinaghiassian/DIAMOnD), and heat_diffusion (based on the following implemantation: https://github.com/idekerlab/heat-diffusion/blob/c434a01913c3a35e2c189955f135f050be246379/heat_diffusion_service.py#L39)

### data
It contains all the necessary data (databases, interactome, ppa) to test the various algorithms on different datasets.
* **biogrid** contains the Homo Sapiens interactome provided by BioGRID (BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt).
* In **diamond_dataset** subfolder you can find the interactome (Interactome.tsv), the gene-disease associations (seeds.tsv) and the list of diseases (diseases.txt) used for the original DIAMOnD paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004120).
* In **gda** there are the gene-disease associations as provided by DisGeNET, both the curated version (curated_gene_disease_associations.tsv) used for the K-Fold Cross Validation and the complete version (all_gene_disease_associations.tsv) used for the Extended Validation.
* In **stringdb** there are the full Homo Sapiens interactome as provided by StringDB (9606.protein.links.full.v11.5.txt) and the aliases file that link the StringDB protein ID with the standard ID (like the one used in BioGRID).
* Finally **disease_file.txt** contains the names of the diseases on which perform the algorithm analysis, you can just edit this file to change the diseases fot the anlysis (IMPORTANT: this file won't be used if you choose to use the *diamond_dataset*, beacuse in this case it will use its own list of diseases).

### disease_analysis
It contains the results of the analysis of the disease network attributes for different datasets. They were computed in multiple settings: using _only the disease genes_, using the _disease genes + first neighbors in the interactome_ and using _only the first neighbors of the disease genes_.

### parsed_results
It contains the performance results for each algorithm and for each dataset.

* _average_results_ contains the scores averaged over all the diseases
* _disease_results_ contains the scores for each disease.

The scores are computed for four different metrics: precision, recall, f1 and ndcg; and for two type of validations (K-Fold and Extended). Please read the PROCONSUL paper for further details.

### plots
Here you can find two plot types:
* _heatmaps_ contains the heatmaps of the comparison between couple of algorithms. At the moment you can find only the comparison between DIAMOnD and PROCONSUL, but you can easily generate the comparison between different algorithms (see: How to run).
* _iteration_score_ contains the plots that represent how the performance (of one metric) evolves as we increase the number of iteration for the different algorithms.

### predicted_genes
It contains the putative disease genes predicted by the different algrithms using different datasets.

### results
It contains the raw performance results obtained by each algorithm for each disease, for each dataset and for each validation type.
They are the starting point from which the _parsed_results_ were computed.

### test
It contains the modules to perform the K-Fold Cross Validation and the Extended Validation.

### tmp
An internal folder used to store the intermiadate results of some computation. No relevan data here.

### utils
It contains varius sets of utility functions used by the main computations.


## Prerequisites
Our code requires the following modules to work properly:

| Module      	| Url                                                     	|
|-------------	|---------------------------------------------------------	|
| Graph Tiger 	| https://graph-tiger.readthedocs.io/en/latest/index.html 	|
| Matplotlib  	| https://matplotlib.org/                                 	|
| NetworkX    	| https://networkx.org/                                   	|
| NumPy       	| https://numpy.org/                                      	|
| Pandas      	| https://pandas.pydata.org/                              	|
| PyTorch     	| https://pytorch.org/                                    	|
| scikit-earn 	| https://scikit-learn.org/stable/                        	|
| SciPy       	| https://scipy.org/                                      	|
| seaborn     	| https://seaborn.pydata.org/                             	|

To install them you can manually install each module by click on the provided url and following the installation procedure or, if you have already installed python3 and pip3 on your computer, use the provided __requiriments.txt__ file with:
```
pip3 install -r requirements.txt
```
Alternatively, if you prefer to use a virtual _Conda_ environment, you can, __after creating the virtual environment__, activate the environment and install the modules through the conda installer:
```
(base) $ conda activate -n env_name
(env_name) $ conda install --file requirements.txt
(env_name) $ pip3 install -r requirements.txt (_since not all the modules are provided by conda__)
```

__IMPORTANT:__ Before running the code you need to __downlad the missing data files__. 
In fact, due to the limitations of github we have not been able to provide all the necessary files. Now we will explain you where to find those missing files and where to place them.

| Missing file  | Download link | Where to place it |
|-------------	|-------------- |------------------ |
| BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3  | https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.204/| Download BIOGRID-ORGANISM-4.4.204.tab3.zip, extract it, copy BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3 and paste it inside ```data/biogrid```|
| adacency_matrix.npy (precomputed adjacency matrix used for network analysis)| | ```data/biogrid```  |
| distance_matrix.npy (precomputed distance matrix used for network analysis)| | ```data/biogrid```  |
| all_gene_disease_associations.tvs | https://www.disgenet.org/downloads  | ```data/gda```  |



## How to run
