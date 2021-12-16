import torch.utils.data as data
import os
import torch
import json
from PIL import Image
from torchvision import transforms
import math
import glob

from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from utils import ROOT_PATH

name_to_class_ids_file = os.path.join(ROOT_PATH, 'transformer/image_name_to_class_id_and_name.json')
# params of dataset
INPUT_SIZE = (3, 224, 224)
INTERPOLATION = 'bicubic'
DEFAULT_CROP_PCT = 0.875 # 0.9, 1.0
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Imagenet 21k
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def params(model_name):
    if model_name in ['vit_deit_base_distilled_patch16_224','levit_256','pit_b_224','cait_s24_224','convit_base', 'visformer_small', 'deit_base_distilled_patch16_224']:
        params = {'mean': IMAGENET_DEFAULT_MEAN,
                  'std': IMAGENET_DEFAULT_STD,
                  'interpolation': INTERPOLATION,
                  'crop_pct':0.9}
        # set the crop_pct as the constant value
        # if model_name == 'convit_base':
        #     params['crop_pct'] = 0.875
        # elif model_name == 'cait_s24_224':
        #     params['crop_pct'] = 1.0
        # else:
        #     params['crop_pct'] = 0.9
    else:
        params = {'mean': IMAGENET_INCEPTION_MEAN,
                  'std': IMAGENET_INCEPTION_STD,
                  'interpolation': INTERPOLATION,
                  'crop_pct': 0.9}
    return params

def transforms_imagenet_wo_resize(params):

    tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(params['mean']),
                     std=torch.tensor(params['std']))
        ]
    return transforms.Compose(tfl)


class AdvDataset(data.Dataset):
    def __init__(self, model_name, adv_path):
        self.transform = transforms_imagenet_wo_resize(params(model_name))
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        print ('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0]+'.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        
        with open(name_to_class_ids_file, 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        image_name = path.split('/')[-1]
        # deal with image
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id, class_name, image_name