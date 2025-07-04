import csv
import os


# Path to the directory
directory_path = '/media/patricija/Pat/gif_data/features'

# List all files in the directory (excluding folders)
files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
# Find matches between fileAdapted and rows in the third column
matches = []
for file in files:
    fileAdapted = file.replace('.pt', '')

    # Path to the TSV file
    tsv_file_path = '/home/patricija/Desktop/GifModelRe/tgif-v1.0.tsv'

    # Read the TSV file and extract the third column
    with open(tsv_file_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            if len(row) > 0:
                gif_id = row[0].split('/')[-1].replace('.gif', '')
                if gif_id == fileAdapted:
                    matches.append(row)
                    # Path to the output CSV file
                    output_csv_path = '/home/patricija/Desktop/GifModelRe/matching_captions.csv'

                    # Write matches to the CSV file in the specified format
                    with open(output_csv_path, 'w', newline='') as output_csv:
                        csv_writer = csv.writer(output_csv)
                        for match in matches:
                            gif_url = match[0]
                            caption = match[1]
                            gif_id = gif_url.split('/')[-1].replace('.gif', '')
                            formatted_row = [gif_url, f"<start> {caption} <end>", gif_id]
                            csv_writer.writerow(formatted_row)