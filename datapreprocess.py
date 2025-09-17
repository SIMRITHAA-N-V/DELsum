#!/usr/bin/python3
import argparse
import sys, os
import nltk
import json
from sklearn.model_selection import train_test_split
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Folder containing documents to be summarized')
parser.add_argument('--prep_path', type=str, help='Path to store the prepared data in json format')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing (default: 0.2)')
parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
parser.add_argument('--stratify', action='store_true', help='Use stratified sampling to maintain class distribution')

args = parser.parse_args()

BASEPATH = args.data_path
writepath = args.prep_path
separator = "\t"

FILES = []
FILES2 = os.listdir(BASEPATH)
for f in FILES2:
    FILES.append(f)

DATA_FILES = {}

for F in FILES:
    ifname = os.path.join(BASEPATH, F)
    print(f'Processing {F}...')
    fp = open(ifname, 'r')
    dic = {}
    
    for l in fp:
        try:
            wl = l.split(separator)
            CL = wl[1].strip(' \t\n\r')
            TEXT = wl[0].strip(' \t\n\r')
            TEXT = TEXT.replace("sino noindex makedatabase footer start url", "")
            
            if TEXT:
                if CL in dic:
                    dic[CL].append(TEXT)
                else:
                    dic[CL] = [TEXT]
        except Exception as e:
            print(f"Error processing line in {F}: {e}")
    
    fp.close()
    
    # Prepare data with NLTK processing
    f_d = {}
    for cl, sentences in dic.items():
        temp = []
        for s in sentences:
            tokens = nltk.word_tokenize(s)
            t = (s, tokens, nltk.pos_tag(tokens))
            temp.append(t)
        f_d[cl] = temp
    
    DATA_FILES[F.split('.txt')[0].strip(' \t\n\r')] = f_d
    print(f'Complete {F}')

# Prepare data for train/test split
def prepare_for_split(data_files):
    """Convert nested dict structure to flat list for train/test splitting"""
    all_samples = []
    labels = []
    file_sources = []
    
    for file_name, file_data in data_files.items():
        for class_label, samples in file_data.items():
            for sample in samples:
                all_samples.append(sample)
                labels.append(class_label)
                file_sources.append(file_name)
    
    return all_samples, labels, file_sources

def split_data_by_class(data_files, test_size=0.2, random_state=42):
    """Split data maintaining class distribution within each file"""
    train_data = {}
    test_data = {}
    
    for file_name, file_data in data_files.items():
        train_data[file_name] = {}
        test_data[file_name] = {}
        
        for class_label, samples in file_data.items():
            if len(samples) == 1:
                # If only one sample, put it in training
                train_data[file_name][class_label] = samples
                test_data[file_name][class_label] = []
            else:
                # Split samples for this class
                train_samples, test_samples = train_test_split(
                    samples, 
                    test_size=test_size, 
                    random_state=random_state
                )
                train_data[file_name][class_label] = train_samples
                test_data[file_name][class_label] = test_samples
    
    return train_data, test_data

def split_data_globally(data_files, test_size=0.2, random_state=42, stratify=True):
    """Split all data globally across files"""
    all_samples, labels, file_sources = prepare_for_split(data_files)
    
    if stratify and len(set(labels)) > 1:
        # Stratified split to maintain class distribution
        indices = list(range(len(all_samples)))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
    else:
        # Regular random split
        indices = list(range(len(all_samples)))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
    
    # Reconstruct data structure for train and test
    train_data = {}
    test_data = {}
    
    for idx in train_idx:
        file_name = file_sources[idx]
        label = labels[idx]
        sample = all_samples[idx]
        
        if file_name not in train_data:
            train_data[file_name] = {}
        if label not in train_data[file_name]:
            train_data[file_name][label] = []
        train_data[file_name][label].append(sample)
    
    for idx in test_idx:
        file_name = file_sources[idx]
        label = labels[idx]
        sample = all_samples[idx]
        
        if file_name not in test_data:
            test_data[file_name] = {}
        if label not in test_data[file_name]:
            test_data[file_name][label] = []
        test_data[file_name][label].append(sample)
    
    return train_data, test_data

# Perform the split
if args.stratify:
    train_data, test_data = split_data_globally(DATA_FILES, args.test_size, args.random_state, stratify=True)
    print("Used stratified global split")
else:
    train_data, test_data = split_data_by_class(DATA_FILES, args.test_size, args.random_state)
    print("Used per-class split within each file")

# Print statistics
def print_stats(data, name):
    total_samples = 0
    class_counts = {}
    
    for file_name, file_data in data.items():
        for class_label, samples in file_data.items():
            total_samples += len(samples)
            if class_label not in class_counts:
                class_counts[class_label] = 0
            class_counts[class_label] += len(samples)
    
    print(f"\n{name} Data Statistics:")
    print(f"Total samples: {total_samples}")
    print("Class distribution:")
    for class_label, count in sorted(class_counts.items()):
        print(f"  {class_label}: {count} samples ({count/total_samples*100:.1f}%)")

print_stats(train_data, "Training")
print_stats(test_data, "Test")

# Save the split data
output_data = {
    'train': train_data,
    'test': test_data,
    'metadata': {
        'test_size': args.test_size,
        'random_state': args.random_state,
        'stratified': args.stratify,
        'total_files': len(DATA_FILES)
    }
}

# Save to files
with open(os.path.join(writepath, 'prepared_data_split.json'), 'w') as f:
    json.dump(output_data, f, indent=4)

# Also save train and test separately for convenience
with open(os.path.join(writepath, 'train_data.json'), 'w') as f:
    json.dump(train_data, f, indent=4)

with open(os.path.join(writepath, 'test_data.json'), 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"\nData saved to:")
print(f"  Combined: {os.path.join(writepath, 'prepared_data_split.json')}")
print(f"  Training: {os.path.join(writepath, 'train_data.json')}")
print(f"  Testing: {os.path.join(writepath, 'test_data.json')}")