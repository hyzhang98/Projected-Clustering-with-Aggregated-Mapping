# Towards Projected Clustering with Aggregated Mapping


This repository is our implementation of 

[Hongyuan Zhang, Yanan Zhu, and Xuelong Li, "Towards Projected Clustering with Aggregated Mapping," *IEEE Transactions on Image Processing*, DOI: 10.1109/TIP.2023.3274989, 2023](https://ieeexplore.ieee.org/document/10179275).



The core motivation to **investigate how deep clustering with explicit aggregation (e.g. GNN) and implicit aggregation (traditional deep clustering) works**. We aim to *uncover the underlying essence and design a more direct model (via explicitly introduction the aggregation operation)*. 

The source code consists of two parts: 

- The code of linear case is implemented by MATLAB. The source file is *Projected_Clustering_Adaptive_Aggregation.m*. The meaning of various parameters can be found in the code comments. 

- The code of a simple deep extension with GNN can be found in the folder *Simple Deep Extension*.  

 &nbsp;
   

If you have issues, please email:

hyzhang98@gmail.com or hyzhang98@mail.nwpu.edu.cn.



### Requirements 

MATLAB R2019b

pytorch 1.10.0

scikit-learn 0.21.3

numpy 1.16.5

## Citation

```
@article{PojectedClustering,
  author={Zhang, Hongyuan and Zhu, Yanan and Li, Xuelong},
  journal={IEEE Transactions on Image Processing}, 
  title={Towards Projected Clustering with Aggregated Mapping}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3274989}
}

```

