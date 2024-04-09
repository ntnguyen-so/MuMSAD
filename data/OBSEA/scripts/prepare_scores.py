import os
import shutil

def is_obsea_data(dataset_name):
    try:
        int(dataset_name[:4])
    except ValueError:
        return False
    return True

def copy_scores_data(src_path, dest_path):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            # Check if the current file is a docker-algorithm-scores.csv
            if file == 'docker-algorithm-scores.csv':
                # Extracting parts of the file path
                parts = root.split(os.sep)
                try:
                    # Assuming the algorithm name and dataset date are always in fixed positions
                    algorithm_name = parts[-5]
                    dataset_date = parts[-2]

                    # move only obsea data which starts with year
                    if not is_obsea_data(dataset_date):
                        continue

                    dataset_date = dataset_date.split('_')[0]
                    
                    # Destination directory path
                    dest_dir = os.path.join(dest_path, algorithm_name)
                    # Ensure the destination directory exists
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Full path for the destination file
                    dest_file_path = os.path.join(dest_dir, f"{dataset_date}.out")
                    
                    # Full path of the current file
                    src_file_path = os.path.join(root, file)
                    
                    # Copy and rename the file
                    shutil.copy(src_file_path, dest_file_path)
                    print(f"Copied: {src_file_path} to {dest_file_path}")
                except IndexError:
                    print("Path structure is different than expected. Skipping file:", file)

if __name__ == "__main__":
    src_path = '/mnt/c/Arbeid/Github_Repo/TimeEval_work/results/'  
    dest_path = '/mnt/c/Arbeid/Github_Repo/MSAD_work/data/OBSEA/scores/'
    copy_scores_data(src_path, dest_path)
