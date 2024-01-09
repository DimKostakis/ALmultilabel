import os
import glob
import pickle
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Define the directory path where your .pkl files are located
directory = '/content/drive/MyDrive/multiresults/'
# Initialize an empty DataFrame to store the data
data = pd.DataFrame(columns=['Query Strategy', 'Estimator', 'F1-Macro'])

# Iterate through all .pkl files in the directory
for file_path in glob.glob(directory + '*.pkl'):
    # Load the data from the .pkl file
    with open(file_path, 'rb') as file:
        pkl_data = pickle.load(file)

    # Extract the relevant information from the file name
    file_name = os.path.basename(file_path)
    dataset,estimator,query_strategy,it,qr,lratio= file_name.split('_')[1:]

    # Extract the f1-macro scores from the loaded data
    f1_macro_scores = pkl_data['f1-macro']

    # Calculate the mean f1-macro score rounded to 2 decimal digits
    f1_macro_mean =sum(f1_macro_scores) / len(f1_macro_scores)

    # Append the data to the DataFrame
    data = data.append({'Query Strategy': query_strategy,
                        'Estimator': estimator,
                        'Iter':it,
                        'Dataset':dataset,
                        'qr':qr,
                        'lratio':lratio,
                        'F1-Macro': f1_macro_mean},
                        ignore_index=True)

# Filter the data based on the specified dataset, iteration, query ratio, and labeling ratio
filtered_data = data[(data['Dataset'] == 'emotions') & (data['Iter'] == 'it=5') & (data['qr'] == 'qr=20') & (data['lratio'] == 'lratio=20.pkl')]

# Find the initial f1-macro as the mean of f1-macros with qr=0 and lratio=20
initial_f1_macro = data[(data['Dataset'] == 'emotions') & (data['qr'] == 'qr=0') & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['F1-Macro'].mean()

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Query Strategy', 'Estimator', 'F1-Macro Increase'])

# Calculate the f1-macro increase for each combination of query strategy and estimator
for query_strategy in filtered_data['Query Strategy'].unique():
    for estimator in filtered_data['Estimator'].unique():
        # Filter the data for the current combination of query strategy and estimator
        subset_data = filtered_data[(filtered_data['Query Strategy'] == query_strategy) & (filtered_data['Estimator'] == estimator)]
        initial_f1_macro = data[(data['Dataset'] == 'emotions') & (data['qr'] == 'qr=0')& (data['Query Strategy']==query_strategy) & (data['Estimator']==estimator) & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['F1-Macro'].mean()
        # Calculate the f1-macro increase by subtracting the initial f1-macro from the final f1-macro
        f1_macro_final = subset_data['F1-Macro'].iloc[-1]
        f1_macro_increase = round((f1_macro_final - initial_f1_macro) / initial_f1_macro * 100, 2)
                # Append the results to the DataFrame
        results_df = results_df.append({'Query Strategy': query_strategy,
                                        'Estimator': estimator,
                                        'F1-Macro Increase': f1_macro_increase},
                                       ignore_index=True)
