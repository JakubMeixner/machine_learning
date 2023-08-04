import csv
import ast

def csv_to_list_of_lists(csv_file, delimiter=','):
    data = []
    with open(csv_file, newline='') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            # Remove quotes and parse elements using ast.literal_eval
            cleaned_row = [ast.literal_eval(element) for element in row]
            data.append(cleaned_row)

    return data

def write_output_to_file(output_file, data):
    with open(output_file, 'w') as file:
        file.write(str(data))

# Example usage for semicolon-separated CSV file:
csv_file = 'Zeszyt1.csv'
list_of_lists = csv_to_list_of_lists(csv_file, delimiter=';')

# Example usage for writing the output to a text file named "output.txt":
output_file = 'output.txt'
write_output_to_file(output_file, list_of_lists)
