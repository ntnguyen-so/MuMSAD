import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import shutil
import statistics

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

def concat_results(directories):
    dfs = list()
    
    # Iterate through execution dates
    for directory in directories:
        for exec_folder in os.listdir(directory):
            for file in os.listdir(os.path.join(directory, exec_folder)):
                if file == 'results.csv':
                    filepath = os.path.join(directory, exec_folder, file)
                    df = pd.read_csv(filepath)
                    df = remove_similar_duplicates(df)
                    df['result_path'] = filepath
                    dfs.append(df)
    
    return pd.concat([df.reset_index(drop=True) for df in dfs], ignore_index=True)

def get_best_run_f1MINUSfnr(all_results):
    obsea_results = all_results[(all_results['collection'] == 'OBSEA') | (all_results['collection'] == 'OBSEA_2')]
    obsea_results = obsea_results[(obsea_results['BestF1Score'].notna()) & (obsea_results['FalseNegativeRate'].notna())]
    obsea_results['collection'][obsea_results['collection'] == 'OBSEA_2'] = 'OBSEA'
    obsea_results['Recommendation_ACC'] =  obsea_results.apply(lambda row: max(row['BestF1Score'] - row['FalseNegativeRate'], 0), axis=1)
    obsea_results = obsea_results.sort_values(by='Recommendation_ACC', ascending=False)
    obsea_results = obsea_results.drop_duplicates(subset=['algorithm', 'collection', 'dataset'], keep='first')
    return obsea_results

def get_best_run(all_results):
    obsea_results = all_results[(all_results['collection'] == 'OBSEA') | (all_results['collection'] == 'OBSEA_2')]
    obsea_results = obsea_results[obsea_results['PR_AUC'].notna()]
    obsea_results['collection'][obsea_results['collection'] == 'OBSEA_2'] = 'OBSEA'
    obsea_results = obsea_results.sort_values(by='PR_AUC', ascending=False)
    obsea_results = obsea_results.drop_duplicates(subset=['algorithm', 'collection', 'dataset'], keep='first')
    return obsea_results

def refresh_content(path):
    # Iterate over all the files and subdirectories within the specified directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Delete the file
            os.remove(file_path)
        for dir in dirs:
            # Construct the full directory path
            dir_path = os.path.join(root, dir)
            # Delete the directory and its contents recursively
            shutil.rmtree(dir_path, ignore_errors=True)


def copy_data(MSAD_root_path, TimeEval_root_path):
    dest_path = MSAD_root_path + 'data/OBSEA/data/OBSEA'
    os.makedirs(dest_path, exist_ok=True)
    refresh_content(dest_path)
    src_path = TimeEval_root_path + '/work_data/processed/multivariate/OBSEA'
    for data_file in os.listdir(src_path):
        if 'test' in data_file:
            df = pd.read_csv(src_path + '/' + data_file)
            df.drop(df.columns[0], axis=1, inplace=True)
            df.to_csv(dest_path + '/' + data_file.split('.')[0] + '.out', index=False, header=False)

def copy_scores(MSAD_root_path, obsea_results):
    dest_path = MSAD_root_path + 'data/OBSEA/scores/OBSEA'
    refresh_content(dest_path)
    for row_idx in range(len(obsea_results)):
        try:
            result_path = '/'.join(obsea_results.iloc[row_idx, obsea_results.columns.get_loc('result_path')].split('/')[:-1])
            algorithm = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('algorithm')]
            if algorithm not in ad2use:
                continue
            hyper_params_id = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('hyper_params_id')]
            collection = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('collection')]
            dataset = obsea_results.iloc[row_idx, obsea_results.columns.get_loc('dataset')]
            if "202" not in dataset:
                continue
            repetition = str(obsea_results.iloc[row_idx, obsea_results.columns.get_loc('repetitions')])
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
    refresh_content(path_to_metrics)
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
            
def copy_metrics_2(MSAD_root_path, obsea_results):
    path_to_metrics = MSAD_root_path + 'data/OBSEA/metrics/'
    refresh_content(path_to_metrics)
    path2metric = '/home/t/00_work/TimeEval_work_results/01_metric/'
    for ad_method in ad2use:
        metrics_dir_alg = path_to_metrics + '/' + ad_method
        os.makedirs(metrics_dir_alg, exist_ok=True)
        src_path = path2metric + ad_method + '.csv'
        desc_metric_path = path_to_metrics + ad_method + '/PR_AUC.csv'
        shutil.copy(src_path, desc_metric_path)
        
def copy_scores_2(MSAD_root_path, obsea_results):
    path_to_scores = MSAD_root_path + 'data/OBSEA/scores/OBSEA'
    org_path2scores = '/home/t/00_work/TimeEval_work_results/00_scores/'
    
    src_folder = org_path2scores
    dest_folder = path_to_scores
    
    # Use shutil.copytree to copy all folders and files
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dest_item = os.path.join(dest_folder, item)
        
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
            # print(f"Copied folder: {src_item} -> {dest_item}")
            
    change_file_extensions(dest_folder)
        
def change_file_extensions(folder_path, old_ext='.csv', new_ext='.out'):
    # Traverse through all files in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Check if the file has the old extension
            if file_name.endswith(old_ext):
                # Construct full file path
                old_file_path = os.path.join(root, file_name)
                # Create new file name by replacing the extension
                new_file_name = file_name.replace(old_ext, new_ext)
                new_file_path = os.path.join(root, new_file_name)
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')

# Specify the folder path here
folder_path = '/path/to/your/folder'
change_file_extensions(folder_path)


def dataprep_useMSAD_noretrain(TimeEval_results_path):
    directory = [TimeEval_results_path + '/transfer_learning_results/' + 'MSAD_trained20_infer2122']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_noMSAD(TimeEval_results_path):
    directory = [TimeEval_results_path + 'best4avgens_2122']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    obsea_results.to_csv('results_noMSAD.csv', index=False)
    return obsea_results

def dataprep_useMSAD_retrain1year(TimeEval_results_path):
    directory = [TimeEval_results_path + 'MSAD_trained20_inferredtrained21_infer22', TimeEval_results_path + 'MSAD_trained20_infer2122']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    obsea_results.to_csv('results_useMSAD_retrain1y.csv', index=False)
    return obsea_results

def dataprep_useMSAD_retrain6month(TimeEval_results_path):
    directory = [TimeEval_results_path + 'MSAD_trained20_infer2122', TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_infer2H21',
                 TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_inferredtrained2H21_infer1H22', 
                 TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_inferredtrained2H21_inferredtrained1H22_infer2H22']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    obsea_results.to_csv('results_useMSAD_retrain6m.csv', index=False)
    return obsea_results

def dataprep_totrainMSAD_20(TimeEval_results_path):
    directory = [TimeEval_results_path + 'training_data_MSAD_2020']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_totrainMSAD_2021(TimeEval_results_path):
    directory = [TimeEval_results_path + 'training_data_MSAD_2020', TimeEval_results_path + 'for_Oracle']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_totrainMSAD_202122(TimeEval_results_path):
    directory = [TimeEval_results_path + 'training_data_MSAD_2020', TimeEval_results_path + 'for_Oracle', TimeEval_results_path + 'for_Oracle3', TimeEval_results_path + 'for_Oracle4', TimeEval_results_path + 'for_Oracle23_1']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_totrainMSAD_23(TimeEval_results_path):
    directory = [TimeEval_results_path + 'for_Oracle23_1', TimeEval_results_path + 'for_Oracle23_2']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_totrainMSAD_20212223(TimeEval_results_path):
    directory = [TimeEval_results_path + 'best4avg_2022',
                TimeEval_results_path + 'best4avgens_2122',
                TimeEval_results_path + 'for_Oracle',
                TimeEval_results_path + 'for_Oracle23_1',
                TimeEval_results_path + 'for_Oracle23_2',
                TimeEval_results_path + 'for_Oracle3',
                TimeEval_results_path + 'for_Oracle4',
                TimeEval_results_path + 'for_Oracle5',
                TimeEval_results_path + 'imputed_training_data_MSAD_2020',
                TimeEval_results_path + 'MSAD_trained20_infer2122',
                TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_infer2H21',
                TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_inferredtrained2H21_infer1H22',
                TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_inferredtrained2H21_inferredtrained1H22_infer2H22',
                TimeEval_results_path + 'MSAD_trained20_inferredtrained21_infer22',
                #TimeEval_results_path + 'multihead',
                TimeEval_results_path + 'RBF_add',
                TimeEval_results_path + 'ReachSubsea_unsupervised',
                TimeEval_results_path + 'training_data_MSAD_2020',
                TimeEval_results_path + 'for_Oracle_240724',
                TimeEval_results_path + 'transfer_learning_results']

    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results



def dataprep_Oracle(TimeEval_results_path):
    directory = [TimeEval_results_path + 'for_Oracle']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)
    return obsea_results

def dataprep_multihead(TimeEval_results_path):
    directory = [TimeEval_results_path + 'multihead/trained20_infer2122']
    all_results = concat_results(directory)
    obsea_results = get_best_run(all_results)
    #obsea_results.to_csv('results_useMSAD_noretrain.csv', index=False)multihead/trained20_infer2122
    return obsea_results
    

if __name__ == "__main__":
    MSAD_root_path = '../../../'
    TimeEval_working_path = MSAD_root_path + '../TimeEval_work/'
    TimeEval_results_path = MSAD_root_path + '../TimeEval_work_results/'
    metrics = ['PR_AUC']

    if False:
        #### START: trained 20, inferredtrained1H21, inferredtrained2H21 to infer 1H22    
        directory = [TimeEval_results_path + 'training_data_MSAD_2020', TimeEval_results_path + 'MSAD_trained20_infer2122']
        all_results = concat_results(directory)
        obsea_results = get_best_run(all_results)
        obsea_results.to_csv('results_trained20_infer2122.csv', index=False)
        obsea_results_1H21 = obsea_results[obsea_results['dataset'] < '2021-07-01']
        obsea_results_1H21.to_csv('results_trained20_infer1H21.csv', index=False)

        directory = [TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_infer2H21']
        all_results = concat_results(directory)
        obsea_results_2H21 = get_best_run(all_results)
        obsea_results = pd.concat([df.reset_index(drop=True) for df in [obsea_results_1H21, obsea_results_2H21]], ignore_index=True)
        obsea_results.to_csv('MSAD_trained20_inferredtrained1H21_infer2H21.csv', index=False)

        directory = [TimeEval_results_path + 'MSAD_trained20_inferredtrained1H21_inferredtrained2H21_infer1H22']
        all_results = concat_results(directory)
        obsea_results_1H22 = get_best_run(all_results)
        obsea_results = pd.concat([df.reset_index(drop=True) for df in [obsea_results, obsea_results_1H22]], ignore_index=True)
        obsea_results.to_csv('MSAD_trained20_inferredtrained1H21_inferredtrained2H21_infer1H22.csv', index=False)
        #### END: trained 20, inferredtrained1H21, inferredtrained2H21 to infer 1H22  

    ad2use = sorted(['AutoEncoder (AE)', 
                          'CBLOF',
                          'COPOD', 
                          'DeepAnT',
                          'DenoisingAutoEncoder (DAE)',
                          'EncDec-AD',
                          'HBOS',
                          'Hybrid KNN',
                          'LOF',
                          'PCC', 
                          'RobustPCA',
                          'Random Black Forest (RR)', 
                          'Torsk', ])
    #obsea_results = dataprep_noMSAD(TimeEval_results_path)
    #obsea_results = dataprep_useMSAD_noretrain(TimeEval_results_path)
    #obsea_results = dataprep_useMSAD_retrain1year(TimeEval_results_path)
    obsea_results = pd.read_csv('/home/t/00_work/TimeEval_work_results/to_save.csv') #dataprep_totrainMSAD_20212223(TimeEval_results_path)
    copy_data(MSAD_root_path, TimeEval_working_path)
    copy_scores_2(MSAD_root_path, obsea_results)
    copy_metrics_2(MSAD_root_path, obsea_results)


