import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

def remove_similar_duplicates(df):
    # Reverse the DataFrame to prioritize keeping the last occurrence of duplicates
    df = df.iloc[:, ::-1]

    # Find columns with similar names and data
    similar_columns = defaultdict(list)
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if df.columns[i].split('.')[0] == df.columns[j].split('.')[0] and df.iloc[:, i].equals(df.iloc[:, j]):
                similar_columns[df.columns[i]].append(df.columns[j])

    # Filter out the similar duplicates, keep only the first instance of each in the reversed dataframe
    for column, duplicates in similar_columns.items():
        df = df.drop(columns=duplicates)

    # Revert column names by removing suffixes (e.g., .1, .2)
    df.columns = [col.split('.')[0] for col in df.columns]

    # Re-reverse the DataFrame to restore original column order
    return df.iloc[:, ::-1]

def concat_results(directory):
    dfs = list()
    
    # Iterate through execution dates
    for exec_folder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, exec_folder)):
            if file == 'results.csv':
                filepath = os.path.join(directory, exec_folder, file)
                df = pd.read_csv(filepath)
                df = remove_similar_duplicates(df)
                df['result_path'] = filepath
                dfs.append(df)
    
    return pd.concat([df.reset_index(drop=True) for df in dfs], ignore_index=True)

def get_best_run(all_results):
    obsea_results = all_results[(all_results['collection'] == 'OBSEA') | (all_results['collection'] == 'OBSEA_2')]
    obsea_results = obsea_results[(obsea_results['BestF1Score'].notna()) & (obsea_results['FalseNegativeRate'].notna())]
    obsea_results['collection'][obsea_results['collection'] == 'OBSEA_2'] = 'OBSEA'
    obsea_results['Recommendation_ACC'] =  obsea_results.apply(lambda row: max(row['BestF1Score'] - row['FalseNegativeRate'], 0), axis=1)
    obsea_results = obsea_results.sort_values(by='Recommendation_ACC', ascending=False)
    obsea_results = obsea_results.drop_duplicates(subset=['algorithm', 'collection', 'dataset'], keep='first')
    return obsea_results

def copy_data(MSAD_root_path, TimeEval_root_path):
    dest_path = MSAD_root_path + 'data/OBSEA/data/OBSEA'
    os.makedirs(dest_path, exist_ok=True)
    src_path = TimeEval_root_path + '/work_data/processed/multivariate/OBSEA'
    for data_file in os.listdir(src_path):
        if 'test' in data_file:
            df = pd.read_csv(src_path + '/' + data_file)
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv(dest_path + '/' + data_file.split('.')[0] + '.out', index=False, header=False)

def copy_scores(MSAD_root_path, obsea_results):
    dest_path = MSAD_root_path + 'data/OBSEA/scores/OBSEA'
    for row_idx in range(len(obsea_results)):
        try:
            result_path = '/'.join(obsea_results.iloc[row_idx, obsea_results.columns.get_loc('result_path')].split('/')[:-1])
            algorithm = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('algorithm')]
            hyper_params_id = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('hyper_params_id')]
            collection = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('collection')]
            dataset = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('dataset')]
            repetition = str(obsea_results.iloc[row_idx, obsea_results.columns.get_loc('repetition')])
            scores_file_name = 'docker-algorithm-scores.csv'
            src_path = result_path + '/' + algorithm + '/' + hyper_params_id + '/' + collection + '/' + dataset + '/' + repetition + '/' + scores_file_name
            scores_dir_alg = dest_path + '/' + algorithm + '/score'
            if '(AE)' in scores_dir_alg:
                scores_dir_alg = scores_dir_alg.replace(' ', '_')
            os.makedirs(scores_dir_alg, exist_ok=True)
            dest_score_path = scores_dir_alg + '/' + dataset.split('_')[0] + '.out'
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_score_path)
            else:
                src_path = src_path.replace('OBSEA', 'OBSEA_2')
                shutil.copy(src_path, dest_score_path)
        except:
            pass

def copy_metrics(MSAD_root_path, obsea_results):
    path_to_metrics = MSAD_root_path + 'data/OBSEA/metrics'
    metrics = ['BestF1MinusFNR', 'Recommendation_ACC']
    exec_algorithms = obsea_results['algorithm'].unique()
    for exec_algorithm in exec_algorithms:
        for metric in metrics:
            df = obsea_results[(obsea_results['algorithm'] == exec_algorithm)]
            df = df[['dataset', metric]]
            df['dataset'] = df['dataset'].str.strip().str.replace('_unsupervised', '.out')
            df['dataset'] = df['dataset'].apply(lambda x: x + '.out' if not x.endswith('.out') else x)
            df['dataset'] = df['dataset'].apply(lambda x: 'OBSEA/' + x)
            df.set_index('dataset', inplace=True)
            df.index.name = ''
            df = df.groupby(level=0).min()
            df.rename(columns={metric: exec_algorithm}, inplace=True)
            df.dropna(inplace=True)
            metrics_dir_alg = path_to_metrics + '/' + exec_algorithm
            os.makedirs(metrics_dir_alg, exist_ok=True)
            df.to_csv(metrics_dir_alg + '/' + metric + '.csv')

if __name__ == "__main__":
    MSAD_root_path = '../../../'
    TimeEval_root_path = MSAD_root_path + '../TimeEval_work/'
    directory = TimeEval_root_path + 'results'
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    copy_data(MSAD_root_path, TimeEval_root_path)
    copy_scores(MSAD_root_path, obsea_results)
    copy_metrics(MSAD_root_path, obsea_results)

