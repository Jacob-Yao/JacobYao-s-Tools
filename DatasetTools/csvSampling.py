import csv
import os, shutil, time, random, glob
from tqdm import tqdm
import numpy as np


SAMPLE_AMOUNT = 100
EXISTANCE_STRICT = True
SHUFFLE = True
KEEP_NAME = False

CSV_NAME = 'list_train.csv'
OUTPUT_FOLDER = 'SAMPLE_{}_{}'.format(SAMPLE_AMOUNT, time.strftime("%Y%m%d-%H%M%S", time.localtime()))



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
        bak_folder = 'CSV_BACKUP/' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '_sample'
        os.mkdir(bak_folder)
        for f in protect_names:
            if os.path.exists(f):
                shutil.copy(f, os.path.join(bak_folder, f)) 

def prt_hello():
    # path validation
    if not os.path.exists(CSV_NAME):
        raise Exception('The csv file does not exist!')
    # hello
    print('----------->>>>> Dataset CSV Sampling <<<<<-----------')
    print('Will sample \"{}\" pairs of data based on csv file \"{}\".'.format(SAMPLE_AMOUNT, CSV_NAME))
    print('Copy to folder \"{}\".'.format(OUTPUT_FOLDER))
    print('Continue? (yes)')
    my_opt = input()
    if not my_opt=='yes':
        print('------------>>>>> Operation Aborted <<<<<------------')
        quit()
    else:
        print('------------------>>>>> START <<<<<------------------')


def main():
    prt_hello()
    backup_files([CSV_NAME])
    path_list = extract_csv(CSV_NAME)
    assert len(path_list)>0

    os.mkdir(OUTPUT_FOLDER)
    for i in range(len(path_list[0])):
        os.mkdir(OUTPUT_FOLDER+'/'+str(i)) 

    if SHUFFLE:
        random.shuffle(path_list)
    for idx in tqdm(range(SAMPLE_AMOUNT)):
        for file_idx in range(len(path_list[idx])):
            if KEEP_NAME:
                shutil.copy(path_list[idx][file_idx], OUTPUT_FOLDER+'/'+str(file_idx)+'/')
            else:
                shutil.copy(path_list[idx][file_idx], OUTPUT_FOLDER+'/'+str(file_idx)+'/'+
                    '{:0>5d}{}'.format(idx, os.path.splitext(path_list[idx][file_idx])[-1]))

    print('------------->>>>> Sampling Finished <<<<<------------')
    print('Saved in {}'.format(OUTPUT_FOLDER))

if __name__ == '__main__':
    main()
