import csv
import os
from tqdm import tqdm
import numpy as np
from random import shuffle
import cv2
from PIL import Image
import glob


FILE_TYPE_0 = '.png'
FILE_TYPE_1 = '.h5'

EXIST_CHECK = True
TYPE_CHECK = True
SIZE_CHECK = False

path = os.getcwd()
path_list = [['nyu_depth_v1/image', 'nyu_depth_v1/depth'],
            ['nyu_depth_v2/image', 'nyu_depth_v2/depth']]

            
# hello
print('Will generate csv file for NYU-depth V1 & V2 dataset.')
print('The path is set to \" {} \".'.format(path))
print('Search for \"{}\" and \"{}\".'.format(FILE_TYPE_0, FILE_TYPE_1))
print('Continue? (yes)')
my_opt = input()
if not my_opt=='yes':
    print('Operation aborted')
    quit()

# root path check
if not os.path.exists(path):
    raise Exception('Path not exist!')

# start scanning
output_list = []
for idx in range(len(path_list)):
    path_tuple = path_list[idx]
    g = glob.glob(os.path.join(path, path_tuple[0], '*'+FILE_TYPE_0))
    for f in g:  
        f2 = f.replace(path_tuple[0], path_tuple[1]).replace(FILE_TYPE_0, FILE_TYPE_1)
        output_list.append([f, f2])

# Scan report
print('Scan finished! Totally {} pairs.'.format(len(output_list)))
print('Validating file existence...')

# Validation
pbar = tqdm(output_list)
validated_output_list = []
for pathpair in pbar:
    if EXIST_CHECK:
        if not (os.path.exists(pathpair[0]) and os.path.exists(pathpair[1])):
            print('Path not exist: ', pathpair )
            continue 
    if TYPE_CHECK:
        if not (pathpair[0][-len(FILE_TYPE_0):]==FILE_TYPE_0 or pathpair[1][-len(FILE_TYPE_1):]==FILE_TYPE_1):
            print('Path not valid: ', os.path.split(pathpair[0])[-1], os.path.split(pathpair[1])[-1])
            continue 
    if SIZE_CHECK:
        a = Image.open(pathpair[0], mode="r")
        b = Image.open(pathpair[1], mode="r")
        if not a.size==b.size:
            print('Skip size not match: ', os.path.split(pathpair[0])[-1], os.path.split(pathpair[1])[-1])
            continue
    # add to list
    validated_output_list.append(pathpair)
output_list = validated_output_list

# prepare to save
print('Validation finished! Totally {} valid pairs.'.format(len(output_list)))
shuffle(output_list)
train_list = output_list[:int(0.9*len(output_list))]
val_list = output_list[int(0.9*len(output_list)) : int(0.95*len(output_list))]
test_list = output_list[int(0.95*len(output_list)):]

# ask
print('Save? (Enter)')
input()

# save
csvFile = open("list_train.csv", "w")
writer = csv.writer(csvFile)
for pathpair in train_list:
    writer.writerow(pathpair)
csvFile.close()

csvFile = open("list_val.csv", "w")
writer = csv.writer(csvFile)
for pathpair in val_list:
    writer.writerow(pathpair)
csvFile.close()

csvFile = open("list_test.csv", "w")
writer = csv.writer(csvFile)
for pathpair in test_list:
    writer.writerow(pathpair)
csvFile.close()

csvFile = open("list_full.csv", "w")
writer = csv.writer(csvFile)
for pathpair in output_list:
    writer.writerow(pathpair)
csvFile.close()

print('CSV files saved!')