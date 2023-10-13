import os
import argparse
import pandas as pd
from sklearn.model_selection import GroupKFold
import zipfile
from zipfile import ZipFile
import shutil
import re

def parse_clstr(filepath):
    clusters = {}
    current_cluster = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[1])
                clusters[current_cluster] = []
            elif current_cluster is not None:
                fasta_id = line.split('\t')[1].split('>')[1].split('...')[0]
                clusters[current_cluster].append(fasta_id)
    id2cluster = {}
    for cluster, ids in clusters.items():
        for id_ in ids:
            id2cluster[id_] = cluster
    
    return clusters, id2cluster

def check_id_not_in_cluster(df, id2cluster):
    
    if len(id2cluster) == len(df):
        return id2cluster
    for tmp_id in df['tmp_id'].values:
        if tmp_id not in id2cluster:
            id2cluster[tmp_id] = -1
    return id2cluster

def check_sequence_type(sequence):

    # Pattern to match DNA sequence
    dna_pattern = r"^[ACGTN]+$"

    # Pattern to match protein sequence
    protein_pattern = r"^[ACDEFGHIKLMNPQRSTVWYX]{11,}$"

    # Match the sequence against the patterns
    if re.match(dna_pattern, sequence):
        return "DNA"
    elif re.match(protein_pattern, sequence):
        return "Protein"
    else:
        return "Unknown"

def find_sequence_columns(df):
    sequence_columns = []
    for column in df.columns:
        if df[column].dtype == 'object':
            column_type = df[column].head(50).apply(check_sequence_type).unique()
            if len(column_type) == 1 and ("DNA" in column_type or "Protein" in column_type):
                sequence_columns.append(column)

    return sequence_columns

def split_fasta(merged_fasta):
    with open(merged_fasta, 'r') as f:
        data = f.read()
    
    entries = data.strip().split('>')[1:]  # Split entries based on ">"
    
    for entry in entries:
        lines = entry.split('\n')  # Split into header and sequence lines
        header = lines[0].strip()
        sequence = ''.join(lines[1:]).strip()
        output_dir = "filtered_files"
        os.makedirs(output_dir, exist_ok=True)
        # Create a new FASTA file with a unique name based on the header
        output_file = os.path.join(output_dir, f'{header}.fasta')
        
        # Write the sequence entry to the respective FASTA file
        with open(output_file, 'w') as f:
            f.write(f">{header}\n{sequence}")
        
        print(f"Created {output_file} successfully.")


        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="fasta files dir path")
    parser.add_argument("--output_dir", help="output dir")
    parser.add_argument("-m", "--memory", help="", type=int, default=5000)
    parser.add_argument("-t", "--threads", help="threads", type=int, default=2)
    parser.add_argument("-c", "--cluster", type=float, help="sequence identity threshold", default=0.4)
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    args = parse_args()
    
    # Directory path containing the FASTA files
    directory = args.data_path

    # Output file path for the merged FASTA file
    merged_file = 'merged.fasta'

    # List to store the content of all FASTA files
    fasta_content = []
    description_name__dict = {}

    # Iterate through each file in the directory
    with open(merged_file, 'w') as merged_fasta:
        for filename in os.listdir(directory):
            if filename.endswith(".fasta") or filename.endswith(".fa"):
                file_path = os.path.join(directory, filename)
                
                # Read the content of the current FASTA file
                with open(file_path, 'r') as fasta_file:
                    content = fasta_file.readlines()
                    description = content[0].strip().replace(">","")
                    description_name__dict[description] = filename
                    # Write the content to the output file
                    merged_fasta.write(''.join(content)+'\n')
    print(description_name__dict)
    
    
    ## cd-hit
    # -n 5 for thresholds 0.7 ~ 1.0
    # -n 4 for thresholds 0.6 ~ 0.7
    # -n 3 for thresholds 0.5 ~ 0.6
    # -n 2 for thresholds 0.4 ~ 0.5
    
    if args.cluster >= 0.7:
        n = 5
    elif args.cluster >= 0.6:
        n = 4
    elif args.cluster >= 0.5:
        n = 3
    else:
        n = 2

    os.system("chmod 777 ./cdhit/*")

    os.system("echo $LD_LIBRARY_PATH")
    
    # Get the path to the currently active Conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    env_lib_path = os.path.join(conda_prefix, 'lib') # type: ignore

    # Set the LD_LIBRARY_PATH environment variable
    os.environ['LD_LIBRARY_PATH'] = env_lib_path

    # Optional: Print the updated value of LD_LIBRARY_PATH
    print("Updated LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

    cmd = f'./cdhit/cd-hit -i {merged_file} -o temp_c{args.cluster}.fasta -c {args.cluster} -n {n} -d 0 -M {args.memory} -T {args.threads}'
    os.system(cmd)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'dna_similarity<{args.cluster}'
    
    with open(f'temp_c{args.cluster}.fasta', 'r') as f:
        data = f.read()
    
    entries = data.strip().split('>')[1:]  # Split entries based on ">"
    
    for entry in entries:
        lines = entry.split('\n')  # Split into header and sequence lines
        header = lines[0].strip()
        sequence = ''.join(lines[1:]).strip()
        os.makedirs(output_dir, exist_ok=True)
        # Create a new FASTA file with a unique name based on the header
        
        output_file = os.path.join(output_dir, description_name__dict[header])
        
        # Write the sequence entry to the respective FASTA file
        with open(output_file, 'w') as f:
            f.write(f">{header}\n{sequence}")
        
        # print(f"Created {output_file} successfully.")
    

    # # os.remove('./temp.fasta')
    # # os.remove(f'./temp_c{args.cluster}.fasta')
    # # os.remove(f'./temp_c{args.cluster}.fasta.clstr')
    
    
