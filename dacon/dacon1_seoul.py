import pandas as pd
import numpy as np
import os
from glob import glob
import torch
import torch.nn as nn

path = './_data/dacon_seoul/'

label_df = pd.read_csv(path + 'train.csv')
print(label_df.head())

def get_train_data(data_dir):
    img_path_list = []
    label_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.PNG')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
        
    # get label
    label_list.extend(label_df['label'])
                
    return img_path_list, label_list

def get_test_data(data_dir):
    img_path_list = []
    
    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.PNG')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    #print(img_path_list)
    
    return img_path_list


print(label_df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 723 entries, 0 to 722
# Data columns (total 2 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   file_name  723 non-null    object
#  1   label      723 non-null    int64
# dtypes: int64(1), object(1)
# memory usage: 11.4+ KB                         

all_img_path, all_label = get_train_data('./_data/dacon_seoul/train')
test_img_path = get_test_data('./_data/dacon_seoul/test')


all_label[:5]
print(all_label)


