########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : generate_features
#
########################################################################

import numpy as np
import pandas as pd
import argparse
import re
import os

from utils.data_loader import DataLoader

from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.catch22 import Catch22


def generate_features(path):
    """Given a dataset it computes the TSFresh automatically extracted 
    features and saves the new dataset (which does not anymore contain
    time series but tabular data) into one .csv in the folder of the
    original dataset

    :param path: path to the dataset to be converted
    """
    for default_fc_parameters in ['catch22']:# ['minimal', 'efficient']:
        window_size = int(re.search(r'\d+', path).group())

        # Create name of new dataset
        dataset_name = [x for x in path.split('/') if str(window_size) in x][0]
        # new_name = f"TSFRESH_{dataset_name}.csv"
        new_name = f"TSFRESH_{dataset_name}_{default_fc_parameters}.npy"
        new_name_label = f"TSFRESH_{dataset_name}_label_{default_fc_parameters}.npy"
        print(os.path.join(path, new_name))

        # Load datasets
        dataloader = DataLoader(path)
        datasets = dataloader.get_dataset_names()
        # df = dataloader.load_df(datasets)
        data, label = dataloader.load_npy(datasets)
        print(data.shape, label.shape)
        
        # Divide df
        # labels = df.pop("label")
        # x = df.to_numpy()[:, np.newaxis]
        # index = df.index

        # Setup the TSFresh feature extractor (too costly to use any other parameter)
        # fe = TSFreshFeatureExtractor(
            # default_fc_parameters=default_fc_parameters, 
            # show_warnings=False, 
            # n_jobs=-1
        # )
        fe = Catch22(
            catch24=True
        )
        
        # Compute features
        # X_transformed = fe.fit_transform(x)
        X_transformed = fe.fit_transform(data)
        np.save(os.path.join(path, new_name), X_transformed)
        np.save(os.path.join(path, new_name_label), label)
        print('Done, saved to', os.path.join(path, new_name_label))
        # print(X_transformed.shape)

        # Create new dataframe
        # X_transformed.index = index
        # X_transformed = pd.merge(labels, X_transformed, left_index=True, right_index=True)
        
        # Save new features
        # X_transformed.to_csv(os.path.join(path, new_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_features',
        description='Transform a dataset of time series (of equal length) to tabular data\
        with TSFresh'
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use')
    
    args = parser.parse_args()
    generate_features(
        path=args.path, 
    )