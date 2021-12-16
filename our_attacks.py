import torch
import argparse
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
import pickle as pkl

from dataset import AdvDataset
import cnns_method
from utils import BASE_ADV_PATH, ROOT_PATH
import our_method

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack', type=str, default='', help='the name of specific attack method')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--filename_prefix', type=str, default='', help='')
    args = parser.parse_args()
    args.opt_path = os.path.join(BASE_ADV_PATH, 'model_{}-method_{}-{}'.format(args.model_name, args.attack, args.filename_prefix))
    if not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # loading dataset
    dataset = AdvDataset(args.model_name, os.path.join(ROOT_PATH, 'clean_resized_images'))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print (args.attack, args.model_name)
    
    # Attack
    attack_method = getattr(our_method, args.attack)(args.model_name)

    # Main
    all_loss_info = {}
    for batch_idx, batch_data in enumerate(data_loader):
        if batch_idx%10 == 0:
            print ('Ruing batch_idx', batch_idx)
        batch_x = batch_data[0]
        batch_y = batch_data[1]
        batch_name = batch_data[3]

        adv_inps, loss_info = attack_method(batch_x, batch_y)
        attack_method._save_images(adv_inps, batch_name, args.opt_path)
        if loss_info is not None:
            all_loss_info[batch_name] = loss_info
    if loss_info is not None:
        with open(os.path.join(args.opt_path, 'loss_info.json'), 'wb') as opt:
            pkl.dump(all_loss_info, opt)
