import logging
import sys
import src.utils.config as config
import os

config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
model_output_path = "/".join([os.getcwd(),config_catalog['output_file_path']['model_output_path'],str(config_catalog['data_preprocessing']['version'])])
log_path = "/".join([model_output_path,'logs.log'])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)