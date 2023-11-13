import csv
import sys

# Set the encoding for standard output to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Open the CSV file with the appropriate encoding and 'utf-8-sig' to handle the BOM
import pandas as pd

# Read the CSV file using pandas
df = pd.read_csv('VRMYwords.csv', encoding='utf-8-sig')

# Initialize an empty list to store the concatenated strings
concatenated_strings = []

# Iterate through the rows in the DataFrame
for index, row in df.iterrows():
    # Filter out 'nan' values and join the non-empty columns with a space
    non_empty_values = ' '.join(str(cell) for cell in row if not pd.isna(cell))
    
    # Append the non-empty values to the list
    concatenated_strings.append(non_empty_values)

# Now you have a list of concatenated strings with non-empty values
#for string in concatenated_strings:
#    print(string)

#print(concatenated_strings)


# Remove empty strings from the list
filtered_list = [item for item in concatenated_strings if item.strip() != ""]
''''
for string in filtered_list:
    print(string)
'''
new_list=[]
for i in range(len(filtered_list)):
    if filtered_list[i][0]=='R' and filtered_list[i][1]=='_':
        new_list.append(filtered_list[i-2]+filtered_list[i]+'.1')
        new_list.append(filtered_list[i-1]+filtered_list[i]+'.2')
#for i in new_list:
 #   print(i)
