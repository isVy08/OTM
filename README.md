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


  
