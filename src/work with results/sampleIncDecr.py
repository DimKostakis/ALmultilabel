import os
import glob
import pickle
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the directory path where your .pkl files are located
directory = '/content/drive/MyDrive/multiresults/'

# Initialize an empty DataFrame to store the data
data = pd.DataFrame(columns=['Query Strategy', 'Estimator', 'F1-Macro', 'Average Precision', 'Recall Macro', 'F1-Micro', 'Hamming Loss'])

# Iterate through all .pkl files in the directory
for file_path in glob.glob(directory + '*.pkl'):
    # Load the data from the .pkl file
    with open(file_path, 'rb') as file:
        pkl_data = pickle.load(file)

    # Extract the relevant information from the file name
    file_name = os.path.basename(file_path)
    dataset, estimator, query_strategy, it, qr, lratio = file_name.split('_')[1:]

    # Extract the performance metrics from the loaded data
    f1_macro = pkl_data['f1-macro']
    recall_macro = pkl_data['Recall-macro']
    f1_micro = pkl_data['f1-micro']
    hamming_loss = pkl_data['HammingLoss']

    # Calculate the mean values of the performance metrics
    f1_macro_mean = sum(f1_macro) / len(f1_macro)
    recall_macro_mean = sum(recall_macro) / len(recall_macro)
    f1_micro_mean = sum(f1_micro) / len(f1_micro)
    hamming_loss_mean = sum(hamming_loss) / len(hamming_loss)

    # Append the data to the DataFrame
    data = data.append({'Query Strategy': query_strategy,
                        'Estimator': estimator,
                        'Dataset':dataset,
                        'Iter':it,
                        'qr':qr,
                        'lratio':lratio,
                        'F1-Macro': f1_macro_mean,
                        'Recall Macro': recall_macro_mean,
                        'F1-Micro': f1_micro_mean,
                        'Hamming Loss': hamming_loss_mean},
                        ignore_index=True)
db='emotions'
# Filter the data based on the specified dataset, iteration, query ratio, and labeling ratio
filtered_data = data[(data['Dataset'] == db) & (data['Iter'] == 'it=5') & (data['qr'] == 'qr=20') & (data['lratio'] == 'lratio=20.pkl')]



# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Query Strategy',
                                    'Estimator',
                                    '%IncF1Macro',
                                    '%IncRecallMacro',
                                    '%IncF1Micro',
                                    '%DecrHammingLoss'])

# Calculate the performance metric increases/decreases for each combination of query strategy and estimator
for query_strategy in filtered_data['Query Strategy'].unique():
    for estimator in filtered_data['Estimator'].unique():
        # Filter the data for the current combination of query strategy and estimator
        subset_data = filtered_data[(filtered_data['Query Strategy'] == query_strategy) & (filtered_data['Estimator'] == estimator)]

        # Calculate the performance metric increases/decreases by subtracting the initial value from the final value
        f1_macro_final = subset_data['F1-Macro'].iloc[-1]
        recall_macro_final = subset_data['Recall Macro'].iloc[-1]
        f1_micro_final = subset_data['F1-Micro'].iloc[-1]
        hamming_loss_final = subset_data['Hamming Loss'].iloc[-1]

        # Find the initial performance metrics as the mean of metrics with qr=0 and lratio=20
        initial_f1_macro = data[(data['Dataset'] == db) & (data['qr'] == 'qr=0')& (data['Query Strategy']==query_strategy) & (data['Estimator']==estimator) & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['F1-Macro'].mean()
        initial_recall_macro = data[(data['Dataset'] == db) & (data['qr'] == 'qr=0')& (data['Query Strategy']==query_strategy) & (data['Estimator']==estimator) & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['Recall Macro'].mean()
        initial_f1_micro = data[(data['Dataset'] == db) & (data['qr'] == 'qr=0')& (data['Query Strategy']==query_strategy) & (data['Estimator']==estimator) & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['F1-Micro'].mean()
        initial_hamming_loss = data[(data['Dataset'] == db) & (data['qr'] == 'qr=0')& (data['Query Strategy']==query_strategy) & (data['Estimator']==estimator) & (data['Iter'] == 'it=1') & (data['lratio'] == 'lratio=20.pkl')]['Hamming Loss'].mean()


        f1_macro_increase = round((f1_macro_final - initial_f1_macro) / initial_f1_macro * 100, 2)
        average_precision_increase = round((average_precision_final - initial_average_precision) / initial_average_precision * 100, 2)
        recall_macro_increase = round((recall_macro_final - initial_recall_macro) / initial_recall_macro * 100, 2)
        f1_micro_increase = round((f1_micro_final - initial_f1_micro) / initial_f1_micro * 100, 2)
        hamming_loss_decrease = round((initial_hamming_loss - hamming_loss_final) / initial_hamming_loss * 100, 2)

        # Append the results to the DataFrame
        results_df = results_df.append({'Query Strategy': query_strategy,
                                        'Estimator': estimator,
                                        '%IncF1Macro': f1_macro_increase,
                                        '%IncRecallMacro': recall_macro_increase,
                                        '%IncF1Micro': f1_micro_increase,
                                        '%DecrHammingLoss': hamming_loss_decrease},
                                       ignore_index=True)
