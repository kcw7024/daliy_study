import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader  # 학습 및 배치로 모델에 넣어주기 위한 툴
import torchvision.transforms as transforms  # 이미지 변환 툴
import torchvision.datasets as datasets  # 이미지 데이터셋 집합체
import random
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
    img_path_list.sort(key=lambda x: len(x.split('/')[-1].split('.')[0]))

    # get label
    label_list.extend(label_df['label'])

    return img_path_list, label_list


def get_test_data(data_dir):
    img_path_list = []

    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.PNG')))
    img_path_list.sort(key=lambda x: len(x.split('/')[-1].split('.')[0]))
    # print(img_path_list)

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

# all_label[:5]
# print(all_label)

# print(test_img_path[:5])


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Arrange GPU devices starting from 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the GPU 2 to use, 멀티 gpu

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    #device = torch.device("cuda:0")
    print('Device:', device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


CFG = {
    'IMG_SIZE': 128,  # 이미지 사이즈
    'EPOCHS': 50,  # 에포크
    'LEARNING_RATE': 2e-2,  # 학습률
    'BATCH_SIZE': 12,  # 배치사이즈
    'SEED': 41,  # 시드
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):  # 필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index):  # index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        # print(img_path)
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):  # 길이 return
        return len(self.img_path_list)


tempdataset = CustomDataset(all_img_path, all_label, train_mode=False)


plt.imshow(tempdataset.__getitem__(0))
