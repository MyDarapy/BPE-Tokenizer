from datasets import load_dataset
import string
import re

# Replace this with the path to your local file
data_files = {'train': 'C:/Users/Katherine Olowookere/Downloads/clpercentage.jsonl'}

#data_files = {'train': '/mnt/c/Users/Katherine Olowookere/Downloads/clpercentage.jsonl'}

# Load the dataset
data = load_dataset('json', data_files=data_files, split='train')

# Print the first few examples
print(data[0])

#Clean the dataset 
def clean_data(sentence):
  sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # Remove all punctuations 
  sentence = ''.join(char for char in sentence if not char.isdigit()) #Remove digits, only retain characters 
  sentence = re.sub(r'\s+', ' ', sentence).strip() #remove whitespaces 
  return sentence

def process_dataset(dataset):
    return [clean_data(sentence['text']) for sentence in data]

#datasets = dataset.shuffle(seed=65).select(range(300000))
cl_data = process_dataset(data)

import tempfile
import os

# Create a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file_name = temp_file.name

    # Write data to the temporary file
    for item in cl_data:
        encoded_data = list(map(int, item.encode('utf-8')))
        temp_file.write(bytearray(encoded_data))

# Now read from the temporary file
with open(temp_file_name, 'rb') as temp_file:
    all_encoded = [int(byte) for byte in temp_file.read()]

print(all_encoded[:19])

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

vocab_size = 7000 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(all_encoded) 
print(ids[:10])

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx




