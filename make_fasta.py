import csv
import os
import pandas as pd

def write_fasta_file(sequence, file_path):
    with open(file_path, 'w') as file:
        file.write(f'>{file_path}\n')
        file.write(sequence)

def convert_csv_to_fasta(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row
        sequence_column = "seq"  # Replace with the index of your sequence column
        for row, idx in reader:
            sequence = row[sequence_column]
            # file_name = f'{row[sequence_column]}'.replace('/', '_').replace('\\', '_') + '.fasta'
            file_name = idx+ '.fasta'
            write_fasta_file(sequence, file_name)

# Replace 'your_csv_file.csv' with the path to your CSV file
if __name__ == '__main__':
    # Read the CSV file
    data = pd.read_csv('mini_soluprot.csv')

    # Create a directory to store the generated FASTA files
    output_dir = 'fasta_files'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each row in the dataframe
    for index, row in data.iterrows():
        # Extract the sequence and corresponding ID
        sequence = row['seq']
        sequence_id = row['sid']
        
        # Create the contents of the FASTA file
        fasta_content = f'>{sequence_id}\n{sequence}'
        
        # Determine the output file path
        output_file_path = os.path.join(output_dir, f'{index}.fasta')
        
        # Write the FASTA content to the output file
        with open(output_file_path, 'w') as output_file:
            output_file.write(fasta_content)


