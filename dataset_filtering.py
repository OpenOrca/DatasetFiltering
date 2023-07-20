import datasets
import os
from tags import tag_list
import json

BATCH_SIZE = 10000
NUM_PROC = os.cpu_count()
DATASET_PATH = "../datasets/OpenOrca"
DATASET_FILE = "3_5M-GPT3_5-Augmented.parquet"
dataset = datasets.load_dataset(DATASET_PATH, data_files=[DATASET_FILE], split="train")
dataset_size = 1000000

def filter_dataset(dataset):
    print("Filtering the dataset based on the tags")
    df = dataset.to_pandas()
    #removes any duplicate ids or questions
    df.drop_duplicates(subset=['id'], inplace=True, keep='first')
    df.drop_duplicates(subset=['question'], inplace=True, keep='first')
    dataset = datasets.Dataset.from_pandas(df)

    print(dataset)

    def BatchTagFunction(examples):
        batch_size = len(examples['response'])
        keep = [True] * batch_size

        for i in range(batch_size):
            example = {
                'id': examples['id'][i],
                'response': examples['response'][i],
                'system_prompt': examples['system_prompt'][i],
                'question': examples['question'][i],
            }
            for tag in tag_list:
                if tag.evaluate(example):
                    keep[i] = False
                    break

        return keep

    cpu_cores = os.cpu_count() or 1

    dataset = dataset.filter(BatchTagFunction, num_proc=cpu_cores, batch_size=BATCH_SIZE, batched=True)
    print(dataset)
    return dataset

def get_orca_distribution_dataset(dataset):
    print("Creating a dataset with the proper distribution")
    # Load dataset_split.json
    with open('prompts.json') as f:
        prompts = json.load(f)['chuncks']

    dataset_chuncks = []

    def BatchTagFunction(examples):
        batch_size = len(examples['response'])
        keep = [True] * batch_size

        for i in range(batch_size):
            example = {
                'id': examples['id'][i],
                'response': examples['response'][i],
                'system_prompt': examples['system_prompt'][i],
                'question': examples['question'][i],
            }
            if prompt['id'] in example['id'] and example['system_prompt'] == prompt['system_prompt']:
                keep[i] = False

        return keep

    for i, prompt in enumerate(prompts):
        print(f"Running {prompt['id']} {prompt['index']}")
        chunck_size = int(prompt['percentage'] / 100 * dataset_size) + 1
        # Filter dataset
        dataset_chunck = dataset.filter(BatchTagFunction, num_proc=12, batch_size=100000, batched=True)
        # Checks to make sure the dataset chunk is large enough
        if len(dataset_chunck) < chunck_size:
            print(f"{prompt['id']} {prompt['index']}: Missing {chunck_size - len(dataset_chunck)} examples")
            continue
        # Split dataset and add to list
        dataset_chuncks.append(dataset_chunck.select(range(chunck_size))
        )

    # Merge all of the datasets together
    merged_dataset = datasets.concatenate_datasets(dataset_chuncks)
    merged_dataset = merged_dataset.shuffle()

    # Only save if the dataset is larger than the target size
    if len(merged_dataset) > dataset_size:
        print(merged_dataset)
        return merged_dataset
    else:
        print("Dataset is too small, not saving. One of the prompts does not have enough data")

dataset = filter_dataset(dataset)
dataset = get_orca_distribution_dataset(dataset)

if dataset:
    filter_dataset_name = DATASET_FILE.split(".")[0] + "-filtered.parquet"
    dataset.to_parquet(filter_dataset_name)
    print(dataset)
