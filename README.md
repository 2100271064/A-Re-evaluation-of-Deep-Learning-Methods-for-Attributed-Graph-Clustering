A-Re-evaluation-of-Deep-Learning-Methods-for-Attributed-Graph-Clustering
===

## 1. Experimental Environment
All methods are trained on a machine with an Intel(R) Xeon(R) E5-2680 v4 @2.40GHz CPU, 128GB RAM, and four NVIDIA Tesla P100 16G GPUs.
The source code of 13 methods are publicly available(See the github source link below for more details). The performance of the existing methods reported are evaluated using the publicly available source code.
The results of the module evaluation are from our implementations using Python 3.6.7 and Pytorch 1.7.0.

## 2. Datasets

#### 2.1 Clean Raw Datasets
We cleaned all the datasets by removing duplicate nodes, nodes with zero-valued attributes, isolated nodes, and self-loops.<br>
Table 1 shows the statistics of the cleaned datasets. All datasets caputure undirected graphs.<br>

![](https://github.com/2100271064/A-Re-evaluation-of-Deep-Learning-Methods-for-Attributed-Graph-Clustering/blob/main/img/table_1.jpg)

><b>Table Description</b><br>
>`"|V|"` indicates the number of nodes.<br>
>`"|E|"` indicates the number of edges.<br>
>`"#Dimensions"` Indicates node feature dimensions.<br>
>`"#Categories"` indicates the total number of label categories.<br>
>`"#WCCs"` indicates the number of weakly connected components.<br>

The experimental results show that the majority of the nodes in the same label category in these datasets are distant in terms of the graph proximity.
Given the groups of nodes produced by an existing algorithm, current evaluation results only illustrate how similar node attribute values in the same group are but they do not reveal how close in the graph the nodes in the same group are, which does not meet the goal of attribute graph clustering.
Hence, evaluating attributed graph clustering algorithms using these datasets is problematic.

#### 2.2 Proposed New Datasets
We create new benchmark datasets by relabeling the nodes in the connected subgraphs in the existing benchmark dataset.<br>
The nodes in the new datasets are relabeled such that the attributes of the nodes in the same group are similar and the majority of node pairs in the same group are close in terms of graph proximity.<br>

![](https://github.com/2100271064/A-Re-evaluation-of-Deep-Learning-Methods-for-Attributed-Graph-Clustering/blob/main/img/table_2.jpg)

><b>Table Description</b><br>
>The table description is similar to Table 1, except that some datasets have external nodes called `"outliers"`. <br>
>Although these points, which only exist to make the graph structure connected, are included in the new datasets, we should ignore them when evaluate performance.
>In other words, they are learned during the network training, but removed during the evaluation stage.

* Download the complete datasets : [Datasets](https://pan.baidu.com/s/1kq9z_YiRzIoYTMITbgR3sg?pwd=fgh2)

## 3. Re-evaluation Methods

#### 3.1 How To Run
【Make sure you are in the code directory】
```
【AGE】
python ./AGE-master/train_mod.py --gnnlayers 1 --upth_st 0.0011 --upth_ed 0.001 --lowth_st 0.1 --lowth_ed 0.5 --dataset acm
python ./AGE-master/train_mod.py --gnnlayers 3 --upth_st 0.0015 --upth_ed 0.001 --lowth_st 0.1 --lowth_ed 0.5 --dataset citeseer
python ./AGE-master/train_mod.py --gnnlayers 8 --upth_st 0.0110 --upth_ed 0.001 --lowth_st 0.1 --lowth_ed 0.5 --dataset cora
python ./AGE-master/train_mod.py --gnnlayers 3 --upth_st 0.0015 --upth_ed 0.001 --lowth_st 0.1 --lowth_ed 0.5 --dataset dblp
python ./AGE-master/train_mod.py --gnnlayers 1 --upth_st 0.0011 --upth_ed 0.001 --lowth_st 0.1 --lowth_ed 0.5 --dataset wiki
python ./AGE-master/train_mod.py --gnnlayers 35 --upth_st 0.0013 --upth_ed 0.001 --lowth_st 0.7 --lowth_ed 0.8 --dataset pubmed

【SDCN】
python ./SDCN-master/sdcn_mod.py --name xxx
python ./SDCN-master/pretrain_mod.py --name xxx (pretrain)
e.g 
python ./SDCN-master/sdcn_mod.py --name pubmed
python ./SDCN-master/pretrain_mod.py --name pubmed

【CaEGCN】
python ./CaEGCN-main/CaEGCN_mod.py --name xxx
python ./CaEGCN-main/CaEGCN_mod_pubmed.py (In particular, the pubmed dataset is trained in batches)
e.g
python ./CaEGCN-main/CaEGCN_mod.py --name acm

【DAEGC】
python ./DAEGC-main/daegc_mod.py --name xxx
python ./DAEGC-main/daegc_pubmed.py (In particular, the pubmed dataset is trained in batches)
python ./DAEGC-main/pretrain.py --name xxx (pretrain)
python ./DAEGC-main/pretrain_pubmed.py (pretrain; In particular, the pubmed dataset is trained in batches)
e.g
python ./DAEGC-main/daegc_mod.py --name acm
python ./DAEGC-main/pretrain.py --name acm

【AGC-DRR】
python ./AGC-DRR-main/demo_mod.py --name acm --lr 5e-4 --view_lr 1e-4
python ./AGC-DRR-main/demo_mod.py --name citeseer --lr 1e-4 --view_lr 1e-3
python ./AGC-DRR-main/demo_mod.py --name cora --lr 5e-4 --view_lr 1e-4
python ./AGC-DRR-main/demo_mod.py --name dblp --lr 5e-4 --view_lr 1e-4
python ./AGC-DRR-main/demo_mod.py --name wiki --lr 5e-4 --view_lr 1e-4
python ./AGC-DRR-main/demo_mod.py --name pubmed --lr 5e-4 --view_lr 1e-4

【DFCN】
python ./DFCN-master/main_mod.py --name xxx
python -u ./DFCN-master/my_pretrain_gae.py --name xxx (pretrain)
python -u ./DFCN-master/my_pretrain_igae.py --name xxx (pretrain)
python -u ./DFCN-master/my_pretrain_DFCN.py --name xxx (pretrain)
e.g
python ./DFCN-master/main_mod.py --name acm
python -u ./DFCN-master/my_pretrain_gae.py --name acm
python -u ./DFCN-master/my_pretrain_igae.py --name acm
python -u ./DFCN-master/my_pretrain_DFCN.py --name acm

【DCRN】
python ./DCRN-main/main_mod.py --name xxx
python ./DCRN-main-batch/main_mod.py (In particular, the pubmed dataset is trained in batches)
python -u ./DCRN-main/my_pretrain_gae.py --name xxx (pretrain)
python -u ./DCRN-main/my_pretrain_igae.py --name xxx (pretrain)
python -u ./DCRN-main/my_pretrain_DCRN.py --name xxx (pretrain)
python ./DCRN-main-batch/my_pretrain_DCRN.py (pretrain; In particular, the pubmed dataset is trained in batches)
e.g
python ./DCRN-main/main_mod.py --name acm
python -u ./DCRN-main/my_pretrain_gae.py --name pubmed
python -u ./DCRN-main/my_pretrain_igae.py --name pubmed
python -u ./DCRN-main/my_pretrain_DCRN.py --name acm

【CDBNE】
python ./CDBNE-master/run.py --name xxx
python ./CDBNE-master/run_pubmed.py (In particular, the pubmed dataset is trained in batches)
python ./CDBNE-master/pretrain.py --name xxx (pretrain)
python ./CDBNE-master/pretrain_pubmed.py (pretrain; In particular, the pubmed dataset is trained in batches)
e.g
python ./CDBNE-master/run.py --name acm
python ./CDBNE-master/pretrain.py --name acm

【DGC-EFR】
python ./DGC-EFR-master/dgc_efr.py --name xxx
python ./DGC-EFR-master/dgc_efr_pubmed.py --name pubmed (In particular, the pubmed dataset is trained in batches)
python ./DGC-EFR-master/pregae_pubmed.py --name pubmed (pretrain; In particular, the pubmed dataset is trained in batches)
e.g
python ./DGC-EFR-master/dgc_efr.py --name acm
python ./DGC-EFR-master/preae.py --name pubmed
python ./DGC-EFR-master/pregae.py --name acm

【GraphEncoder】
python ./graphencoder-master/graphencoder.py --dataset xxx
e.g
python ./graphencoder-master/graphencoder.py --dataset pubmed

【MGAE】
（Run train.m in Matlab）

【GRACE】
python ./GRACE-main/GRACE_mod.py --dataset xxx
python ./GRACE-main/GRACE_mod.py --dataset pubmed --tol 0.0 --alpha 0.5
e.g
python ./GRACE-main/GRACE_mod.py --dataset acm

【R-GAE】
R-DGAE：python ./R-GAE-master/R-DGAE/main_mod.py --dataset xxx
R-GMM-VGAE：python ./R-GAE-master/R-GMM-VGAE/main_mod.py --dataset xxx
e.g
R-DGAE：python ./R-GAE-master/R-DGAE/main_mod.py --dataset pubmed
R-GMM-VGAE：python ./R-GAE-master/R-GMM-VGAE/main_mod.py --dataset pubmed
```

#### 3.2 Github Source Code Link And Paper
[AGE](https://github.com/thunlp/AGE) : 《Adaptive graph encoder for attributed graph embedding》<br>
[SDCN](https://github.com/bdy9527/SDCN) : 《Structural deep clustering network》<br>
[CaEGCN](https://github.com/huogy/CaEGCN) : 《CaEGCN: Cross-attention fusion based enhanced graph convolutional network for clustering》<br>
[DAEGC](https://github.com/Tiger101010/DAEGC) : 《Attributed graph clustering: A deep attentional embedding approach》<br>
[AGC-DRR](https://github.com/gongleii/AGC-DRR) : 《Attributed Graph Clustering with Dual Redundancy Reduction》<br>
[DFCN](https://github.com/WxTu/DFCN) : 《Deep fusion clustering network》<br>
[DCRN](https://github.com/yueliu1999/DCRN) : 《Deep Graph Clustering via Dual Correlation Reduction》<br>
[CDBNE](https://github.com/xidizxc/CDBNE) : 《Community detection based on unsupervised attributed network embedding》<br>
[DGC-EFR](https://github.com/grcai/DGC-EFR) : 《Deep graph clustering with enhanced feature representations for community detection》<br>
[GraphEncoder](https://github.com/zepx/graphencoder) : 《Learning deep representations for graph clustering》<br>
[MGAE](https://github.com/GRAND-Lab/MGAE) : 《Mgae: Marginalized graph autoencoder for graph clustering》<br>
[GRACE](https://github.com/BarakeelFanseu/GRACE) : 《Grace: A general graph convolution framework for attributed graph clustering》<br>
[R-GAE](https://github.com/nairouz/R-GAE) : 《Rethinking graph auto-encoder models for attributed graph clustering》<br>

## 4. Module Evaluation

#### 4.1 How To Run
【Make sure you are in the code directory】
```
python ./module_evaluation.py main.py --name xxx
e.g
python ./module_evaluation.py main.py --name pubmed
```

* <b>Note</b> <br>
(1) Need to switch to the network model and train function what you want in the "main.py". <br>
(In particular, the pubmed dataset is trained by the "train_pubmed" function.) <br>
(2) Need to open the corresponding code block of the network model and loss function what you want in the "train.py/train_pubmed.py".
