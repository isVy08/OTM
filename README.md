# OTM
This repo includes codes for reproducing the experiments in the paper [Optimal Transport for Structure Learning Under Missing Data](https://arxiv.org/abs/2402.15255)
 accepted at ICML 2024.

## Dependencies
In the project directory, install all the packages in the "requirements.txt".
```
pip install -r requirements.txt
```
The real-world datasets are taken from [Causal Discovery Toolbox](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html) and [Neuropathic Pain Diagnosis Simulator](https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator). 

## Data
The script `config.py` maintains different configurations to generate missingness for various simulation setups. 
`config_id` assigns an index for a combination of configurations. A generated dataset is given a code of format `[sem_type]-[graph_type][config_id]`. 

To generate the datasets, run
```
mkdir dataset
python config.py
```

## Experiments on OTM
For the non-linear case, OTM runs on [DAGMA](https://arxiv.org/abs/2209.08037), which uses Adam optimizer. To run OTM, for example on an MCAR dataset with 10% missing rate (corresponding to `config_id=1`) that follows an ER structure and MLP causal model, run the following command. Create an `output/` directory to save the estimated DAG results. 
```
mkdir output
python main.py 1 ER mlp
```
A dataset will be automatically generated if it has not been done so in the previous step. 

For the linear case, OTM runs on [NOTEARS](https://arxiv.org/abs/1803.01422), which uses L-BFGS-B optimizer. 
```
python linear_solver.py 1 ER
```
## Baseline methods
Use `miss_baselines.py` to run the imputation baselines. It first imputes the missing data and runs DAGMA for non-linear causal discovery and NOTEARs for linear case. 
For example, to experiment with `missforest` imputer in the above setting, run 
```
python miss_baseline.py 1 ER mlp missforest
```
To run MissDAG, refer to `missdag.py`. The codes are taken from [MissDAG repo](https://github.com/ErdunGAO/MissDAG). 

## Citation
If you use the codes or datasets in this repository, please cite our paper.


