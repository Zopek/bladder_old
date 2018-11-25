'''
import csv
import json
import os

file_path = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset'
for file in os.listdir(file_path):
	print file
	csv_path = os.path.join(file_path, file)
	csv_file1 = csv.reader(open(csv_path, 'r'))
	for i in csv_file1:
		print i
'''


'''
filepath = '/DATA3_DB7/data/public/renji_data/labels/label_data.json'
with open(filepath, 'r') as json_file1:
	file = json.loads(json_file1)
	for i in file:
		print i
'''
import numpy as np 

a = [1, 2, 3]
b = np.array(a)

print b