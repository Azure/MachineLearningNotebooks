import csv
import os


output_directory = 'results/'

def init():
    os.makedirs(output_directory, exist_ok=True)


def run(files):
    total = 0
    for input_file in files:
        # "SalesJan2009.1.csv" -> "1"
        file_number = input_file.split('.')[1]
        with open(input_file, 'r') as csvfile:
            rows = csv.reader(csvfile)

            for fields in rows:
                total += float(fields[2])

            result_file_path = os.path.join(output_directory, 'totals-{}.txt'.format(file_number))
            with open(result_file_path) as result_file:
                result_file.write('Total for {}: {}\n'.format(day, total))
