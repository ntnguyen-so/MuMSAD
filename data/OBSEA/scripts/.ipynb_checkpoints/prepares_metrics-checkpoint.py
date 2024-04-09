import os
import pandas as pd

def concat_results(directory):
    # Initialize an empty DataFrame to store concatenated results
    concatenated_df = pd.DataFrame()

    # Iterate through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is named 'results.csv'
            if file == 'results.csv':
                # Construct the full path to the results.csv file
                filepath = os.path.join(root, file)
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(filepath)
                
                # Concatenate the DataFrame to the existing concatenated_df
                concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    
    return concatenated_df

# Specify the directory where the search for results.csv will be performed
directory = '/path/to/your/directory'

# Call the function to concatenate results.csv files
concatenated_results = concat_results(directory)

# Display the concatenated results
print(concatenated_results)
