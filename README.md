# Author: 
Anand K
Senior Data Analyst, QB - CCN
McKinsey & Company
E-Mail: anand_k@mckinsey.com
Phone: +918075969244


# Optimization Engine
This is a python library for solving sequential optimization problems. For faster execution, models are executed parallelly.


# Compatibility
- Python version = 3.9.13
- pandas==2.0.1
- numpy==1.24.3
- pyculiarity==0.0.7
- plotly==5.14.1
- pyyaml==6.0
- openpyxl==3.1.2
- seaborn==0.12.2
- optuna==3.1.1
- shap==0.41.0
- xlrd==2.0.1
- dash==2.9.3


### Optimization technique used:


- Optuna


## Installation
From the python console/terminal, run

```bash
 pip install -r requirements.txt
```

## Usage Steps:
1. In the config --> catalogs folder, Open data_preprocessing.yaml file to enter run_type : 'EA 5' or 'EA 7' as per respective version of the plant, 
      - Benchmark machine compute summary : 10 cores, 64 GB RAM 

2. In the config --> parameters folder, Open optimization_engine.yaml file to enter model configuarations
      - Configure optimization_engine.yaml : optuna_disable_logging (1 to disable show optuna iterations wise logging, 0 to show logging)
      - Configure optimization_engine.yaml : sequence (number of days to start with in 1 cycle to be optimized)
      - Configure optimization_engine.yaml : multiply (number of times the default trails(10) needs to be multiplied for the best sequences with highest revenue)
      - Configure optimization_engine.yaml : drop_per_iteration (factor by which redundant sequences needs to be dropped based on achived revenue)
      - Configure optimization_engine.yaml : min_thresh (number of best sequnces with highest revenue)
      - Configure optimization_engine.yaml : iterations (number of default trails)
      - Configure optimization_engine.yaml : etac_tp (etac throughput)
      - Configure optimization_engine.yaml : function_ (function of highboiler formation to number of days)
      - Configure optimization_engine.yaml : incremental_afterblowdown_kettle (cost impact in 1% drop in kettle level)
      - Configure optimization_engine.yaml : incremental_blowdown_qty (cost impact in 1kg increase in blowdown quantity)
      - Configure optimization_engine.yaml : high_boiler_cutoff (high boiler cut-off for shutdown or sequence break)
      - Configure optimization_engine.yaml : contribution_margin (contribution margin on a normal day)
      - Configure optimization_engine.yaml : shutdown_rev (shutdown cost)
      - Configure optimization_engine.yaml : bdqty_ (range of blow down quantity to be optimized)
      - Configure optimization_engine.yaml : kettle_level_range_u (upper level of kettle in %)
      - Configure optimization_engine.yaml : kettel_level_range_l (lower level of kettle in %)
      - Configure optimization_engine.yaml : kettel_level_max (max kettle level in tonnes)
      - Configure optimization_engine.yaml : n_working_days (number of working days in a year)

      
3. From the python console/terminal, run


```bash
      python3 optimization_pipeline.py
```


5. Results & outputs will be stored in the respective folders 