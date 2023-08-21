from src.modelling.optimize import (
generate_opt_data,
days_to_event,
optimize,
wrapper,
plot_animation
)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


import optuna
from optuna.visualization import plot_edf
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate


from itertools import product
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


import src.utils.config as config
config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
version = str(config_catalog['data_preprocessing']['version']) 
model_output_path = "/".join([os.getcwd(),config_catalog['output_file_path']['model_output_path'],version])
newpath = model_output_path 
processed_data_path =  "/".join([newpath,'processed_data'])
final_results_path =  "/".join([newpath,'final_results'])
plot_summary_path =  "/".join([newpath,'plot_summary'])
# Output Paths
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(processed_data_path)
    os.makedirs(final_results_path)
    os.makedirs(plot_summary_path)
   

optuna_disable_logging = config_parameters['optimize_engineSettings']['optimizer']['optuna_disable_logging']
import src.utils.logger as logger

class optimizer():
    
    all = {}
    

    # Inputs
    increa_afterblowdown_kettle = config_parameters['optimize_engineSettings']['optimizer']['incremental_afterblowdown_kettle']
    increa_blowdown_qty = config_parameters['optimize_engineSettings']['optimizer']['incremental_blowdown_qty']
    function_ =  config_parameters['optimize_engineSettings']['optimizer']['function_']

    if function_ == 'linear' : functions = config_parameters['optimize_engineSettings']['optimizer']['functions']['linear']
    else: functions = config_parameters['optimize_engineSettings']['optimizer']['functions']['non_linear']

    multiply =  config_parameters['optimize_engineSettings']['optimizer']['multiply']
    drop_per_iteration =  config_parameters['optimize_engineSettings']['optimizer']['drop_per_iteration']
    min_thresh =  config_parameters['optimize_engineSettings']['optimizer']['min_thresh']
    iterations =  config_parameters['optimize_engineSettings']['optimizer']['iterations']
    sequence = config_parameters['optimize_engineSettings']['optimizer']['sequence']

    opt_parameters  = config_parameters['optimize_engineSettings']['optimizer']
    logger.logger.info(f" Optimization parameters : function_:{function_, functions} || min_thresh:{min_thresh} || multiply:{multiply} || trials:{iterations} || drop_per_iteration:{drop_per_iteration}")
        

    

    def __init__(self,name,df):

        self.name = name
        self.df = df

        optimizer.all[self.name] = self.df 



    
    @classmethod
    def instantiate_inputs(cls):
        
        # Generate input files
        opt_df = generate_opt_data(optimizer.sequence,processed_data_path)
        opt_df = pd.concat([opt_df,opt_df.loc[3:],opt_df.loc[3:],opt_df.loc[3:],opt_df.loc[3:],opt_df.loc[3:],opt_df.loc[3:]]).reset_index(drop=True)
        cols_to_opt = opt_df.columns.to_list()
        logger.logger.info(f" Sequnce created : data shape - {opt_df.shape}")
        optimizer('opt_df',opt_df), optimizer('cols_to_opt',cols_to_opt)

    
    
    
    @classmethod
    def optimize(cls):
        best_params_dic, study_opt , opt_datafame = wrapper(optuna_disable_logging, optimizer.all['opt_df'], optimizer.all['cols_to_opt'], 
                                                    optimizer.iterations, optimizer.increa_afterblowdown_kettle, optimizer.increa_blowdown_qty, 
                                                    optimizer.opt_parameters, function_ = optimizer.function_, min_thresh = optimizer.min_thresh,
                                                    multiply = optimizer.multiply, drop_per_iteration = optimizer.drop_per_iteration )
        opt_datafame.columns = ['BD qty', 'KL after BD', 'HB', 'NO BD', 'Sequence ID', 'Revenue']
        opt_datafame.to_csv(os.path.join(final_results_path,'optimized_top.csv'),index=False)
        logger.logger.info(f" Optimization dataframe generated : data shape - {opt_datafame.shape}")
        optimizer('opt_datafame', opt_datafame), optimizer('best_params_dic', best_params_dic), optimizer('study_opt', study_opt)

    

    
    @classmethod
    def animate(cls):
        opt_datafame = optimizer.all['opt_datafame']
        study_opt = optimizer.all['study_opt']
        for seq in list(opt_datafame['Sequence ID'].unique()):
            test = opt_datafame[opt_datafame['Sequence ID']== seq]
            test = pd.concat([test[test['Revenue']==x] for x in sorted(list(test['Revenue'].unique()))])

            animation, optimization_history, parallel_coordinate, feature_importance, edf = plot_animation(test,study_opt,seq)

            with open(os.path.join(plot_summary_path,'optimization_summary_'+ seq +'.html'), 'a') as f:
                f.write(animation.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(optimization_history.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(parallel_coordinate.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(feature_importance.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(edf.to_html(full_html=False, include_plotlyjs='cdn'))

        

        logger.logger.info(" Optimization complete")
        logger.logger.info(f" Optimization Run catalogs : {config_catalog}")
        logger.logger.info(f" Optimization Run parameters : {config_parameters}")
        logger.logging.shutdown()
        
    




    def __repr__(self):
        return f"{self.name}, {self.df}"