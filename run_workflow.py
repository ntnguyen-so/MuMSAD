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
from utils.evaluator import Evaluator, load_classifier
from utils.config import detector_names
import os
import math
import subprocess
from sklearn.decomposition import PCA
import time
import traceback
import random
# from utils.evaluator import Evaluator, load_classifier
import pickle

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

def select_AD_model(model_path, model_name, model_parameters_file, window_size, ts_data, top_k, batch_size, is_deep=False):
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
    if is_deep:
        # load model parameters
        model_parameters = json_file(model_parameters_file)
        if 'original_length' in model_parameters:
            model_parameters['original_length'] = window_size * num_dimensions
        if 'timeseries_size' in model_parameters:
            model_parameters['timeseries_size'] = window_size * num_dimensions
        
        # print('model_name', model_name)
        # load model
        model = deep_models[model_name](**model_parameters)    
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to('cuda')
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            model.to('cpu')
        
        preds = model(sequence.float()).argmax(dim=1).tolist()
    else:
        model = load_classifier(model_path)
        preds = model.predict(sequence)    
        preds = list(preds)
        preds = [int(x) for x in preds]
    
    print(preds)
    # Majority voting
    counter = Counter(preds)
    most_k_voted = counter.most_common(top_k)
    top_k_detectors = [x[0] for x in most_k_voted]
    suggested_detectors = [detector_names[d] for d in top_k_detectors]
    
    return top_k_detectors

def read_metadata():
    curr_path = os.getcwd()
    path2metadata = curr_path + '/../../TimeEval_work/work_data/processed/datasets.csv'
    metadata_df = pd.read_csv(path2metadata)
    return metadata_df

def get_dataset_id(metadata_df, data_file):
    index = metadata_df[metadata_df['dataset_name'].str.contains(data_file)].index.values.tolist()
    return [index[0], index[-1]]



if __name__ == "__main__":
    # MSAD_model_base_path = '/mnt/c/Users/ntng/AppData/Roaming/Notepad++/plugins/Config/NppFTP/Cache/ntnguyen@thalia.mi.parisdescartes.fr/home/ntnguyen/MSAD_work/results/weights/'
    MSAD_model_base_paths = ['results/weights/']#, 'results/weights_no-znorm_32/']
    votes_folders = ['results/votes/']
    deep_models_shortform = ['convnet', 'inception', 'resnet', 'sit']
    fe_base = 'catch22'
    for MSAD_model_base_path, votes_folder in zip(MSAD_model_base_paths, votes_folders):
        print(MSAD_model_base_path, votes_folder)
        if 'catch22' in MSAD_model_base_path:
            fe_base = 'catch22'
        for model in sorted(os.listdir(MSAD_model_base_path)):
            print('model:', model)
            is_deep_model = False
            for deep_model_name in deep_models_shortform:
                if deep_model_name in model:
                    is_deep_model = True
                    
            # if is_deep_model:
                # continue
            for saved_model in sorted(os.listdir(MSAD_model_base_path + model)):
                print(MSAD_model_base_path, model, saved_model, is_deep_model)
                try:
                    # model_path='./results/weights/sit_stem_original_32/model_06072024_134554'
                    model_path = MSAD_model_base_path + model + '/' + saved_model
                    model_name=model_path.split('/')[-2].split('_')[0]
                    model_parameters_file='./models/configuration/' + "_".join(model_path.split('/')[-2].split('_')[:-1]) + '.json' # sit_stem_original.json'#sit_linear_patch.json'
                    # print(model_parameters_file)
                    path_to_data = './data/OBSEA/data/OBSEA/'
                    num_k = 4
                    processes_running = list()
                    max_process = 10
                    # metadata_df = read_metadata()
                    curr_path = os.getcwd()
                    data_files = os.listdir(path_to_data)
                    files_checked = list()
                    files_choice = list()
                    #random.shuffle(data_files)
                    data_files = data_files[::1]
                    
                    window_size = int(model_path.split('/')[-2].split('_')[-1])
                    # if window_size != 16:
                        # continue
                            
                    if not is_deep_model:
                        fe_name = 'TSFRESH_' + fe_base + '.pkl'
                        fe_path = './data/OBSEA_' + str(window_size) + '/' + fe_name
                        with open(f'{fe_path}', 'rb') as input:
                            fe = pickle.load(input)
                    
                    
                    for data_file in data_files:
                        print(data_file)
                        if "2022" not in data_file and "_data" not in data_file:
                            continue
                        if "unsupervised" in data_file:
                            continue

                        uploaded_ts = path_to_data + data_file
                        
                        if False: # for PCA (univariate)
                            ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
                            ts_data = ts_data_raw[:, :-1].astype(float)
                            ts_data = PCA(n_components=1).fit_transform(ts_data)
                            # print(ts_data)
                            sequence = z_normalization(ts_data, decimals=7)
                        else:
                            if False: # for MinMaxScaler
                                scaler_min = np.array([27.0415, 3.32207, 12.7718])
                                scaler_max = np.array([38.4214, 5.94431, 27.4442])
                                ts_data_raw = pd.read_csv(uploaded_ts, header=None).to_numpy()
                                ts_data = ts_data_raw[:, :-1].astype(float)
                                ts_data = (ts_data - scaler_min) / (scaler_max - scaler_min)
                            else:
                                np_mean = np.array([37.80610348, 4.94523778, 18.21861115])
                                np_std = np.array([0.18258118, 0.03025344, 0.23778808])
                                ts_data_raw = pd.read_csv(uploaded_ts, header=None).to_numpy()
                                ts_data = ts_data_raw[:, :-1].astype(float)
                                ts_data = (ts_data - np_mean) / (np_std)
                            sequence = ts_data
                            # sequence = z_normalization(ts_data, decimals=7)
                            # print(sequence)
                        
                        # Split timeseries and load to cpu
                        sequence = split_ts(sequence, window_size)#[:, np.newaxis]
                        if is_deep_model:
                            sequence = np.swapaxes(sequence, 1, 2)
                        if not is_deep_model:
                            sequence = np.swapaxes(sequence, 1, 2)
                            sequence = fe.transform(sequence)
                            meanval = np.nanmean(sequence)
                            sequence = np.where(np.isnan(sequence), meanval, sequence)# np.nan_to_num(data)
                        print(sequence.shape)
                        
                        if is_deep_model:
                            if torch.cuda.is_available():
                                sequence = torch.from_numpy(sequence).to('cuda')
                            else:
                                sequence = torch.from_numpy(sequence).to('cpu')
                                
                        pred_detector = select_AD_model(model_name=model_name, 
                                                        model_path=model_path, 
                                                        model_parameters_file=model_parameters_file, 
                                                        window_size=window_size, 
                                                        ts_data=ts_data, 
                                                        top_k=num_k, 
                                                        batch_size=64,
                                                        is_deep=is_deep_model)
                                                        
                        print(pred_detector)
                        files_checked.append('OBSEA/' + data_file)
                        files_choice.append(pred_detector[0])
                        # decisions['OBSEA/' + data_file] = pred_detector[0]
                        
                        
                    decisions = {'files': files_checked, 'choice': files_choice}
                    df = pd.DataFrame.from_dict(decisions)#, columns=['name', 'choice'])
                    # print(df)
                    df.to_csv(votes_folder + '/' + model_path.split('/')[-2] + '.csv')    
                    print('save ' + votes_folder + '/' + model_path.split('/')[-2] + '.csv')
                except:
                    traceback.print_exc()
                    pass