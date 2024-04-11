import pandas as pd
import numpy as np

import argparse
import re
import os
from collections import Counter

import torch

from utils.config import *
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator
from utils.config import detector_names
import os
import math

def split_ts(data, window_size):
	'''Split a timeserie into windows according to window_size.
	If the timeserie can not be divided exactly by the window_size
	then the first window will overlap the second.

	:param data: the timeserie to be segmented
	:param window_size: the size of the windows
	:return data_split: an 2D array of the segmented time series
	'''

	# Compute the modulo
	modulo = data.shape[0] % window_size

	# Compute the number of windows
	k = data[modulo:].shape[0] / window_size
	assert(math.ceil(k) == k)

	# Split the timeserie
	data_split = np.split(data[modulo:], k)
	if modulo != 0:
		data_split.insert(0, list(data[:window_size]))
	data_split = np.asarray(data_split)

	return data_split

def z_normalization(ts, decimals=5):
	ts = (ts - np.mean(ts)) / np.std(ts)
	ts = np.around(ts, decimals=decimals)

	# Test normalization
	assert(
		np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
	), "After normalization it should: mean == 0 and std == 1"

	return ts

def select_AD_model(model_path, model_name, model_parameters_file, window_size, ts_data, top_k, batch_size):
	"""Evaluate a deep learning model on time series data and predict the time series.

    :param model: Preloaded model instance.
	:param model_path: Path to the pretrained model weights.
	:param model_parameters_file: Path to the JSON file containing model parameters.
	:param window_size: the size of the window timeseries will be split to (must align with the model)
	:param model: Preloaded model instance.
	:param ts_data: Time series data
	:param top_k: k AD models yielding the highest acc
	Returns:
	"""
	# load model parameters
	model_parameters = json_file(model_parameters_file)
	if 'original_length' in model_parameters:
		model_parameters['original_length'] = window_size * num_dimensions
	if 'timeseries_size' in model_parameters:
		model_parameters['timeseries_size'] = window_size * num_dimensions
		
    # load model
	model = deep_models[model_name](**model_parameters)	
	if torch.cuda.is_available():
		model.load_state_dict(torch.load(model_path))
		model.eval()
		model.to('cuda')
	else:
		model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
		model.eval()
	
    # Generate predictions
	preds = model(sequence.float()).argmax(dim=1).tolist()
	
    # Majority voting
	counter = Counter(preds)
	most_k_voted = counter.most_common(top_k)
	top_k_detectors = [x[0] for x in most_k_voted]
	suggested_detectors = [detector_names[d] for d in top_k_detectors]
	
	return suggested_detectors


if __name__ == "__main__":
    model_name='resnet'
    model_path='/mnt/c/Arbeid/Github_Repo/MSAD_work/results/weights/resnet_default_16/model_11042024_184057'
    model_parameters_file='models/configuration/resnet_default.json'

    uploaded_ts = '/mnt/c/Arbeid/Github_Repo/MSAD_work/data/OBSEA/data/OBSEA/2020-05-10.out'
    ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
    
    ts_data = ts_data_raw[:, 0].astype(float)
    sequence = z_normalization(ts_data, decimals=5)
    window_size = int(model_path.split('/')[-2].split('_')[-1])
    
	# Split timeseries and load to cpu
    sequence = split_ts(sequence, window_size)[:, np.newaxis]
    sequence = torch.from_numpy(sequence).to('cpu')
    
    pred_detector = select_AD_model(model_name=model_name, 
									model_path=model_path, 
									model_parameters_file=model_parameters_file, 
									window_size=window_size, 
									ts_data=ts_data, 
									top_k=4, 
									batch_size=128)
    print(pred_detector)
    