import warnings
warnings.filterwarnings('ignore')
import src.optimizer as optimizer
import src.utils.logger as logger
import src.utils.config as config
from time import time
import os
import datetime

# Load config
config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
now = str(datetime.datetime.now().replace(second=0, microsecond=0)).replace(' ','-').replace(':','')
version = str(config_catalog['data_preprocessing']['version'])
model_output_path = "/".join([os.getcwd(),config_catalog['output_file_path']['model_output_path'],str(config_catalog['data_preprocessing']['version'])])





def run_pipeline():
    """
    Main pipeline for Running Optimizer
    """
    
    t1 = time()
    logger.logger.info(f" Optimization run started")
    logger.logger.info(" loading data preprocessing pipeline...")
    optimizer.optimizer.instantiate_inputs()
    logger.logger.info(" data preprocessing pipeline complete...")
    
    logger.logger.info(" loading model parameters.")
    logger.logger.info(" optimization started...")
    optimizer.optimizer.optimize()
    logger.logger.info(" model run complete...")
    
    logger.logger.info(" Generating animations and plots...")
    logger.logger.info(" Saving model outputs...")
    optimizer.optimizer.animate()
    logger.logger.info(" Optimization run complete.")
    t2 = time()
    elapsed = (t2 - t1)/60
    logger.logger.info(" Optimization run time is %f minutes." % elapsed)
    os.rename(model_output_path, model_output_path +"_"+now)
    for root, dirs, files in os.walk(model_output_path +"_"+now):
            for name in dirs:
                os.rename(os.path.join(root, name), os.path.join(root, name +"_"+now))
            for i in dirs:
                dirrr = os.path.join(model_output_path +"_"+now,i)
                for root, dirs, files in os.walk(dirrr+"_"+now, topdown=False):
                    for file in files:
                        file_name = os.path.splitext(file)[0]
                        extension = os.path.splitext(file)[1]
                        dir_name = "_"+now
                        os.rename(root+"/"+file,root+"/"+file_name+dir_name+extension)


if __name__ == "__main__":
    logger.logger.info(" Running main pipeline...")
    run_pipeline()