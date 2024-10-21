import pandas as pd
from datasets import Dataset, DatasetDict

def load_data(file_path):
    with open(file_path, 'r') as file:
        data_lines = file.readlines()

    data_dicts = []
    columns = data_lines[0].strip().split()
    for line in data_lines[1:]:
        values = line.strip().split(maxsplit=5)
        row_dict = dict(zip(columns, values))
        data_dicts.append(row_dict)
    
    df = pd.DataFrame(data_dicts)
    return df

# Define the paths to your text files
train_file_path = 'data_origin/train.txt'
val_file_path = 'data_origin/test.txt'
test_file_path = 'data_origin/val.txt'

# Load the data from each file
train_df = load_data(train_file_path)
val_df = load_data(val_file_path)
test_df = load_data(test_file_path)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine the datasets into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Save the DatasetDict to disk
dataset_dict.save_to_disk('data')

print('Dataset structure with train, validation, and test sets has been created and saved to disk.')
