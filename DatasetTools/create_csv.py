import csv
import os, shutil, time, random, glob
from tqdm import tqdm
import numpy as np

# path options
DATASET_NAME = 'nyu_depth_v1&v2'
FILE_TYPE_LIST = ['.png', '.h5']
ROOT_PATH = os.getcwd()
FOLDER_LIST = [['nyu_depth_v1/image', 'nyu_depth_v1/depth'],
                ['nyu_depth_v2/image', 'nyu_depth_v2/depth']]

# pipeline controll
EXIST_CHECK = True
TYPE_CHECK = False
CONTENT_CHECK = False

# display & output options
ABS_PATH = False
SHUFFLE = True
DISP_WARNS = True
FILE_PROTECTION = True


    
def pre_scan():
    output_list = []
    # search folder pairs
    for folder_tuple in tqdm(FOLDER_LIST):
        g = glob.glob(os.path.join(ROOT_PATH, folder_tuple[0], '*'+FILE_TYPE_LIST[0]))
        for f in g:  
            f2 = f.replace(folder_tuple[0], folder_tuple[1]).replace(FILE_TYPE_LIST[0], FILE_TYPE_LIST[1])
            if ABS_PATH:
                output_list.append([f, f2])
            else:
                output_list.append([f.replace(ROOT_PATH+'/',''), f2.replace(ROOT_PATH+'/','')])
    print('Scan finished! Totally {} pairs.'.format(len(output_list)))
    return output_list


# Validation
def validation(output_list):
    print('Validating file existence...')
    validated_output_list = []
    for pathpair in tqdm(output_list):
        if EXIST_CHECK:
            # Existancy check for all element
            if not all(os.path.exists(x) for x in pathpair):
                if DISP_WARNS: print('Skip path not exist: ', pathpair )
                continue 
        if TYPE_CHECK:
            # Type check for all element; 1 to 1, not applicable to some situations
            if not all(pathpair[i][-len(FILE_TYPE_LIST[i]):]==FILE_TYPE_LIST[i] for i in range(len(FILE_TYPE_LIST))):
                if DISP_WARNS: print('Skip path not valid: ', pathpair)
                continue 
        if CONTENT_CHECK:
            result = content_check(pathpair)
            if not result:
                if DISP_WARNS: print('Skip content check fail: ', pathpair)
                continue
        # add to list
        validated_output_list.append(pathpair)
    print('Validation finished! Totally {} valid pairs.'.format(len(validated_output_list)))
    return validated_output_list


# content check
def content_check(pathpair):
    # import cv2
    # a = cv2.imread(pathpair[0])
    # b = cv2.imread(pathpair[1])
    # if a.shape==b.shape:
    #     return True
    # else:
    #     return False
    return True

# save csv files
def save_files(output_list):
    # file backup
    protect_names = ['list_train.csv', 'list_val.csv', 'list_test.csv', 'list_full.csv']
    if not os.path.exists('CSV_BACKUP'):
        os.mkdir('CSV_BACKUP')
    if any(os.path.exists(x) for x in protect_names):
        bak_folder = 'CSV_BACKUP/' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '_create'
        os.mkdir(bak_folder)
        for f in protect_names:
            if os.path.exists(f):
                shutil.move(f, os.path.join(bak_folder, f)) 

    # shuffle
    if SHUFFLE:
        random.shuffle(output_list)

    # save
    train_list = output_list[:int(0.9*len(output_list))]
    val_list = output_list[int(0.9*len(output_list)) : int(0.95*len(output_list))]
    test_list = output_list[int(0.95*len(output_list)):]

    with open("list_train.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        for pathpair in train_list:
            writer.writerow(pathpair)

    with open("list_val.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        for pathpair in val_list:
            writer.writerow(pathpair)

    with open("list_test.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        for pathpair in test_list:
            writer.writerow(pathpair)

    with open("list_full.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        for pathpair in output_list:
            writer.writerow(pathpair)

    # protect
    if FILE_PROTECTION:
        os.chmod("list_full.csv", 0o555)

    print('--------------->>>>> File Saved <<<<<----------------')


def prt_hello():
    # path validation
    if not os.path.exists(ROOT_PATH):
        raise Exception('Root path not exist!')
    # hello
    print('---------->>>>> Dataset CSV Generator <<<<<----------')
    print('Will generate {}csv file for \"{}\" dataset.'.format(
        '\"protected\" ' if FILE_PROTECTION else '', DATASET_NAME))
    print('The root path is set to \"{}\".'.format(ROOT_PATH))
    print('Search for \"{}\" with {}\"{}\" paths.'.format(
        FILE_TYPE_LIST, 
        '\"shuffled\" ' if SHUFFLE else '', 
        'absolute' if ABS_PATH else 'relative'))
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
    path_list = pre_scan()
    print('-------------->>>>> Scan Finished <<<<<--------------')
    path_list = validation(path_list)
    prt_asksave()
    save_files(path_list)

if __name__ == '__main__':
    main()