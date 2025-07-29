# SAMamba: Structure-Aware Mamba for Ethereum Fraud Detection


Official implementation of SAMamba: Structure-Aware Mamba for Ethereum Fraud Detection.


## Network Architecture

![Overview](./figures/overview.png)

Comparison of SAMamba with standard GNN-based methods. Existing methods use GNN’s message passing mechanism similar to the 1-WL
algorithm to encode the subtree structure around a target node, and a layer of GNN aggregates one-hop neighbors around the target node. SAMamba encodes
a subgraph around a target node, allowing transactions between neighbors to be perceived, and broadens the receptive field to the full-graph perspective to
gather information about high-order neighborhoods.




## Data Description
### lw_AIG dataset

Size: 5732 graphs, divided into four sub-datasets, Eth-ICO, Eth-mining, Eth-Exchange, Eth-Phish&Hack, with 146, 130, 386, 5070 graphs respectively.

Challenge: The challenge on the lw-AIG dataset is to perform binary classification on all graphs in each sub-dataset.

In the lw-AIG dataset, nodes serve as representations of accounts, while edges encode transaction data and contract callback information. The dataset is stratified into four distinct subsets according to account identity labels derived from the Label Word Cloud of the Ethereum blockchain explorer: ICO wallets, mining accounts, exchange accounts, and phishing&hack accounts. For each subset, graph sampling is performed based on varying edge attributes—Amount, Times, or avgAmount—yielding three distinct graph types. Each graph is anchored to a target account, with the label of the target account assigned as the graph’s label. This setup facilitates the learning of a mapping function that associates graph patterns with account identity labels, enabling the modeling of structural and behavioral characteristics specific to different account types within the Ethereum ecosystem.


Preparation:  Download lw_AIG dataset in PYG format from this [page](https://jjzhou.notion.site/Ethident-Data-861199675dc7454eb36157eeee09cf5b) and place it under the path 'datasets_lw_AIG/data/', and transform the datasets:
```
python data_transform.py
```

### EPTransNet dataset

Size: The original graph includes 2,973,382 nodes and 13,551,214 edges, and the number of labeled phishing nodes is 1,157.

Challenge: The challenge on the EPTransNet dataset is to perform binary classification for each node.

Given the large scale of the original EPTransNet graph, random walk algorithm is employed to sample subgraphs of sizes 30,000, 40,000, and 50,000 nodes, respectively. During the sampling phase, the graph is treated as undirected, with directional edges directly converted to undirected ones to ensure undirected neighbor relationships. The node classification task is subsequently performed on these three subgraphs with distinct sizes.

Preparation:  Download EPTransNet dataset from this [page](https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network/data) and place it under the path 'datasets_phish_scam/', and specify --sample-size to sample the sub-dataset:
```
python sample.py --sample-size 30000
```


## Benchmark
### lw_AIG dataset
 Summary of classification performance in terms of Micro F1-score on the lw-AIG dataset. The highest performance is marked in bold, and the second best performance is underlined.
![Benchmark](./figures/benchmark1.png)


### EPTransNet dataset

Summary of classification performance on the EPTransNet dataset, including precision, recall and F1 score. The highest performance is marked in bold, and the second best performance is underlined. OOM stands for Out of Memory.
![Benchmark](./figures/benchmark2.png)

## Usage

### lw_AIG dataset
```
python train_lw_aig.py
```

### EPTransNet dataset
```
python train_phish_scam.py
```







