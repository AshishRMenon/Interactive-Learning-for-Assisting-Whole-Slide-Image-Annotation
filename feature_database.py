from custom_datasets import get_triplet_dataset,foldered_dataset,dataset_from_embeddings_metadata,dataset_from_embeddings_list
from get_emb_ret import get_model_embeddings_from_dataloader,get_model_embeddings_from_embeddings,get_ranked_images
import numpy as np
import pandas as pd
import argparse
import h5py
import os
import itertools
from itertools import combinations 
import copy
import random
import time
import shutil
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method
import torch.multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method
# import encoding
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def save_features(dataset,model,save_folder,part=0,save_as="query"):
    dataloader_list = []
    embeddings_list = []
    metadata_list = []
    step = min(len(dataset),10000)
    cnt=0
    save_pth = save_folder
    os.makedirs(save_pth,exist_ok=True)
    if len(dataset)>1:
        for i in range(0,len(dataset),step):
            if (i+step)<=len(dataset):
                dataset_part = Subset(dataset, np.arange(i,i+step))
            else:
                dataset_part = Subset(dataset, np.arange(i,len(dataset)))
            dataloader_part = torch.utils.data.DataLoader(dataset_part,batch_size=256,shuffle=False,num_workers=0)
            embeddings_part,meta_data_part = get_model_embeddings_from_dataloader(model,dataloader_part)
            # embeddings_part = np.concatenate(np.array(embeddings_part))
            print("saving embeddings and annot of size",embeddings_part.shape,meta_data_part.shape)
            with h5py.File(save_pth + '/{}_part_{}_{}_emb.h5'.format(save_as,part,i), 'w') as f:
                    f.create_dataset('embed', data=embeddings_part)
            meta_data_part.to_csv(save_pth + '/{}_part_{}_{}_emb.csv'.format(save_as,part,i),index=False)
    else:
        pass
    

def get_base_model_trainable_model(resnet_model,layer=4,sub_layer=1):
    if sub_layer==0:
        trainable = nn.Sequential(*list(resnet_model.children())[-((6-layer+1)):],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)
        base = nn.Sequential(*list(resnet_model.children())[:-((6-layer+1))])    
    if sub_layer==1:
        trainable = nn.Sequential(getattr(resnet_model, 'layer{}'.format(layer))[sub_layer],*list(resnet_model.children())[-((6-layer)):],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)
        base = nn.Sequential(*list(resnet_model.children())[:-(6-layer+1)],getattr(resnet_model, 'layer{}'.format(layer))[sub_layer-1])
    return base,trainable




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for Running Fb Retrieval from a search DB and a query DB")
    parser.add_argument("--dataroot", type=str, required=True, help="Root location of images")
    parser.add_argument("--save_dir", type=str, required=True, help="Save location")
    parser.add_argument("--label_set", type=str, nargs='+', default=None, help="Class Names")
    parser.add_argument("--load_model_path", type=str, default = None)
    parser.add_argument("--use_fc", type=int, default =0)
    parser.add_argument("--label_pos", type=int, default = 2)
    parser.add_argument("--dataset", type=str, default = 'CRC')
    parser.add_argument("--trainable_layer", type=int, default = 4)
    parser.add_argument("--trainable_sub_layer", type=int, default =1)
    args = parser.parse_args()
    img_size=224
    T = {}
    # T['train'] = transforms.Compose([
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    #    ])

    # T['val'] = transforms.Compose([
    #    transforms.CenterCrop(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    #    ])
    
    mean = {}
    std = {}
    mean['CRC'] = [0.7407, 0.5331, 0.7060]
    mean['ICIAR'] = [0.7552, 0.5978, 0.7233]
    std['CRC'] = [0.2048, 0.2673, 0.1872]
    std['ICIAR'] = [0.1753, 0.2293, 0.1631]
    mean['patch_camelyon_train'] = [0.7008, 0.5384, 0.6916]
    std['patch_camelyon_train'] = [0.2531, 0.3060, 0.2307]

    mean['patch_camelyon_val'] = [0.6975, 0.5348, 0.6880]
    std['patch_camelyon_val'] = [0.2539, 0.3068, 0.2320]
    T['train'] = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
       transforms.Normalize(mean[args.dataset], std[args.dataset])
       ])

    T['val'] = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
       transforms.Normalize(mean[args.dataset], std[args.dataset])
       ])
    
    
    if args.label_set is not None:
        labels_database = args.label_set
    else:
        labels_database = os.listdir(args.dataroot)

    print(labels_database)
    
    dataset_full = foldered_dataset(root= args.dataroot , Transform = T['train'],given_classes = args.label_set, label_pos=args.label_pos)
    print(len(dataset_full.img_list))
    print(len(dataset_full))
    search_indices = []
    
    
    dataset_search1 = Subset(dataset_full, np.arange(0,len(dataset_full)//2))
    dataset_search2 = Subset(dataset_full, np.arange(len(dataset_full)//2,len(dataset_full)))
    resnet_model = models.resnet18(pretrained = True)
    resnet_model.fc = Identity()
    base_model_part1,trainable_model = get_base_model_trainable_model(resnet_model,layer=args.trainable_layer,sub_layer=args.trainable_sub_layer)


        
    base_model_part1 = nn.DataParallel(base_model_part1).to(device)
    if args.load_model_path is not None:
        ckpt = torch.load(args.load_model_path)
        base_model_part1.load_state_dict(ckpt)
        print("loaded_model")
        
    save_folder = args.save_dir
    save_features(dataset_search1,base_model_part1,save_folder,part=1,save_as="search_remaining")
    
    save_features(dataset_search2,base_model_part1,save_folder,part=2,save_as="search_remaining")
    
    

