import os
import logging
import functools
from termcolor import colored

#@functools.lru_cache() - since config is dict, it's unhashable
def create_logger(output_dir, config):
    # create logger 
    logger = logging.getLogger(config["module_name"])
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # prevent the log messages to the parent logger
    
    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    
    # create console handler            
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    
    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, config["log_name"]))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    return logger