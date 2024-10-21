import os
from googleapiclient import discovery, errors
from datasets import load_from_disk
import time
import pandas as pd
from tqdm import tqdm

API_KEY = 'Write your api key'
client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

def get_toxicity_score(text, max_retries=3, retry_delay=2):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['ko']
    }
    
    retries = 0
    while retries < max_retries:
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            score = response['attributeScores']['TOXICITY']['summaryScore']['value']
            return score
        except (ConnectionResetError, errors.HttpError) as e:
            print(f"Error: {e}, retrying {retries + 1}/{max_retries}...")
            retries += 1
            time.sleep(retry_delay)
    
    # If all retries fail, return a default value or handle appropriately
    print("Max retries reached, skipping this text.")
    return None

def save_toxic_contents(toxic_contents, batch_num):
    df_toxic = pd.DataFrame(toxic_contents)
    df_toxic.to_csv(f'toxic_contents_batch_{batch_num}.csv', index=False)
    print(f"{batch_num} saved to 'toxic_contents_batch_{batch_num}.csv'")

def main():

    dataset_dict = load_from_disk('data')

    toxic_contents = []
    batch_size = 1000
    batch_num = 0
    toxic_counter = 0

    for split_name, dataset in dataset_dict.items():
        for example in tqdm(dataset, desc="Analyzing toxicity"):
            score = get_toxicity_score(example['content'])
            if score is not None and score >= 0.5:
                example['toxic_score'] = score
                toxic_contents.append(example)
                toxic_counter += 1

                if toxic_counter % batch_size == 0:
                    batch_num += 1
                    save_toxic_contents(toxic_contents, batch_num)
                    toxic_contents = [] 
            
            time.sleep(1)  # To avoid hitting API rate limits

    if toxic_contents:
        batch_num += 1
        save_toxic_contents(toxic_contents, batch_num)

if __name__ == "__main__":
    main()
