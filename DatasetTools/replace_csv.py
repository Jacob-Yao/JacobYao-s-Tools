import csv
import os, shutil, time, random, glob
from tqdm import tqdm
import numpy as np


CSV_NAMES = ['list_train.csv', 'list_val.csv', 'list_test.csv', 'list_full.csv']
REPLACE_FROM = ['my_path1', 'my_path2']
REPLACE_TO = ['your_path1','your_path2']

def extract_csv(csv_file):
    path_list = []
    with open(csv_file, "r") as csvFile:
        reader = csv.reader(csvFile)
        for pathpair in reader:
            path_list.append(pathpair)
    return path_list


def backup_files(protect_names):
    if not os.path.exists('CSV_BACKUP'):
        os.mkdir('CSV_BACKUP')
    if any(os.path.exists(x) for x in protect_names):
        bak_folder = 'CSV_BACKUP/' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
        os.mkdir(bak_folder)
        for f in protect_names:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(bak_folder, f)) 

# save csv files
def save_file(file_name, output_list):
    with open(file_name, "w") as csvFile:
        writer = csv.writer(csvFile)
        for pathpair in output_list:
            writer.writerow(pathpair)

    print('--------------->>>>> File Saved <<<<<----------------')


def prt_hello():
    # path validation
    if not all(os.path.exists(x) for x in CSV_NAMES):
        raise Exception('At least one of the csv names does not exist!')
    assert len(REPLACE_FROM)==len(REPLACE_TO)
    # hello
    print('---------->>>>> Dataset CSV Replacement <<<<<----------')
    print('Will operate on csv files: \"{}\".'.format(CSV_NAMES))
    for idx in range(len(REPLACE_FROM)):
        print('Will replace \"{}\"'.format(REPLACE_FROM[idx]))
        print('          to \"{}\"'.format(REPLACE_TO[idx]))
    print('Continue? (yes)')
    my_opt = input()
    if not my_opt=='yes':
        print('------------>>>>> Operation Aborted <<<<<------------')
        quit()
    else:
        print('------------------>>>>> START <<<<<------------------')

def prt_asksave():
    print('Save files? (save)')
    my_opt = input()
    if not my_opt=='save':
        print('------------>>>>> Operation Aborted <<<<<------------')
        quit()
    else:
        print('----------------->>>>> SAVING <<<<<------------------')



def main():
    prt_hello()
    backup_files(CSV_NAMES)
    for file_name in CSV_NAMES:
        path_list = extract_csv(file_name)
        for i in range(len(path_list)):
            for j in range(len(path_list[i])):
                for k in range(len(REPLACE_FROM)):
                    path_list[i][j] = path_list[i][j].replace(REPLACE_FROM[k], REPLACE_TO[k])
        os.remove(file_name)
        save_file(file_name, path_list)

    print('----------->>>>> Replacement Finished <<<<<-----------')

if __name__ == '__main__':
    main()