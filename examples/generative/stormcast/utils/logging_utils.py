import os
import logging

_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def config_logger(log_level=logging.INFO):
  logging.basicConfig(format=_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):

  if not os.path.exists(os.path.dirname(log_filename)):
    os.makedirs(os.path.dirname(log_filename))

  if logger_name is not None:
    log = logging.getLogger(logger_name)
  else:
    log = logging.getLogger()

  fh = logging.FileHandler(log_filename)
  fh.setLevel(log_level)
  fh.setFormatter(logging.Formatter(_format))
  log.addHandler(fh)

def log_versions():
  import torch
  import subprocess

  try:
    logging.info('--------------- Versions ---------------')
    logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
    logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
    logging.info('Torch: ' + str(torch.__version__))
    logging.info('----------------------------------------')
  except:
    pass
