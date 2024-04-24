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
import subprocess
from sklearn.decomposition import PCA
import time
import random

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
	data_split = data_split.reshape((data_split.shape[0], -1))

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
	print(preds)
	# Majority voting
	counter = Counter(preds)
	most_k_voted = counter.most_common(top_k)
	top_k_detectors = [x[0] for x in most_k_voted]
	suggested_detectors = [detector_names[d] for d in top_k_detectors]
	
	return top_k_detectors

def read_metadata():
	curr_path = os.getcwd()
	path2metadata = curr_path + '/../TimeEval_work/work_data/processed/datasets.csv'
	metadata_df = pd.read_csv(path2metadata)
	return metadata_df

def get_dataset_id(metadata_df, data_file):
	index = metadata_df[metadata_df['dataset_name'].str.contains(data_file)].index.values.tolist()
	return [index[0], index[-1]]



if __name__ == "__main__":
	model_name='resnet'
	model_path='./results/weights/resnet_default_128/model_21042024_212839'
	model_parameters_file='./models/configuration/resnet_default.json'#sit_linear_patch.json'
	path_to_data = './data/OBSEA/data/OBSEA/'
	num_k = 4
	processes_running = list()
	max_process = 10
	metadata_df = read_metadata()
	curr_path = os.getcwd()
	data_files = os.listdir(path_to_data)
	#random.shuffle(data_files)
	data_files = data_files[::1]
	for data_file in data_files:
		print(data_file)
		if "2022-07" not in data_file and "2022-08" not in data_file and "2022-09" not in data_file and "2022-10" not in data_file and "2022-11" not in data_file and "2022-12" not in data_file: # and "2022" not in data_file:
			continue
		if "unsupervised" in data_file:
			continue

		uploaded_ts = path_to_data + data_file
		ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
		ts_data = ts_data_raw[:, :-1].astype(float)
		ts_data = PCA(n_components=1).fit_transform(ts_data)
		#print(ts_data)
		sequence = z_normalization(ts_data, decimals=7)
		window_size = int(model_path.split('/')[-2].split('_')[-1])
		
		# Split timeseries and load to cpu
		sequence = split_ts(sequence, window_size)[:, np.newaxis]
		sequence = torch.from_numpy(sequence).to('cpu')
		
		pred_detector = select_AD_model(model_name=model_name, 
										model_path=model_path, 
										model_parameters_file=model_parameters_file, 
										window_size=window_size, 
										ts_data=ts_data, 
										top_k=num_k, 
										batch_size=256)
		
		for i in range(min(len(pred_detector), num_k)):
			program = 'python3.8'			
			timeeval_path = '../TimeEval_work/'
			os.chdir(timeeval_path)
			path_to_script = '2024-04-01_OBSEA-data.py'
			dataset = data_file.split('.')[0]
			dataset_index = get_dataset_id(metadata_df, dataset)
			num_retry = 3
			command = program + ' ' + path_to_script + ' --dataset_range "(' + str(dataset_index[0]) + ',' + str(dataset_index[1]+1) \
				              + ')" --repetitions ' + str(num_retry) + ' --alg_select ' + str(pred_detector[i])
			print(command)
			process = subprocess.Popen(command, shell=True)
			processes_running.append(process)
			time.sleep(60)
			while len(processes_running) >= max_process:
				for process in processes_running:
					if process.poll() is not None:
						processes_running.remove(process)

				if len(processes_running) < max_process:
					break

			os.chdir(curr_path)

