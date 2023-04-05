import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache() # test and train all need to call it
def create_logger(output_dir, name=''):
    # create logger 
    logger = logging.getLogger(name)
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
    file_handler = logging.FileHandler(output_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    return logger