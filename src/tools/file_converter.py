import csv

def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open(txt_file, 'w') as txt_file:
            for row in csv_reader:
                txt_file.write(','.join(row) + '\n')