import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import models
import os
import numpy as np
import pandas as pd
import glob
from PIL import Image
from PIL import ImageFile
import h5py
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
import time
import random
import copy
def extract_embeddings_csv(db_file,db_csv_file):
    db = h5py.File(db_file, 'r')
    db_csv = pd.read_csv(db_csv_file)
    db_embeddings = np.array(db["embed"])
    return db_embeddings,db_csv


class foldered_dataset():
    def __init__(self,root,given_classes=None, samples_per_class= None, binary_labelling=False, label_of_interest=None,common_class_name = None,Transform=None,label_pos=2):
        super().__init__()
        self.img_list,self.labels = self.get_file_list(root,label_pos,given_classes, samples_per_class,binary_labelling=binary_labelling,label_of_interest=label_of_interest,common_class_name=common_class_name)
        self.classes = list(set(self.labels))
        self.classes.sort()
        self.Transform = Transform
        self.get_metadata = False
        self.label_pos = label_pos
        self.binary_labelling = binary_labelling
        self.label_of_interest = label_of_interest
        self.common_class_name = common_class_name
        
    def __getitem__(self,index):
        img_sample = Image.open(self.img_list[index]).convert('RGB')
        address = self.img_list[index]
        name = self.img_list[index].split('/')[-1]
        class_name = self.labels[index]
        if self.binary_labelling:
            if class_name in self.label_of_interest or class_name == self.common_class_name:
                class_id=1
                class_name_modified = self.common_class_name
            else:
                class_id=0
                class_name_modified = class_name
        else:
            class_id = self.classes.index(class_name)
            class_name_modified = class_name
        
        if self.get_metadata:
            return class_id,address,class_name
        else:
            if self.Transform:
                img_transformed = self.Transform(img_sample)
            return img_transformed,class_id,address,class_name_modified

    def get_file_list(self,root_dir,label_pos,given_classes=None,samples_per_class=None,extensions=['.png','.tif','.TIFF','.JPEG', '.jpg'],binary_labelling=False,label_of_interest=None,common_class_name=None):
        file_list = []
        class_name_list = []
        for root, directories, filenames in os.walk(root_dir): #walk through entire directory
            if given_classes is None:
                given_classes = directories   # if given_classes is None assume the given_classes to be all of classes seen in directories
            if samples_per_class is not None: # if samples per class is given
                filenames = filenames[:samples_per_class]
            for filename in filenames:
                class_name = os.path.join(root, filename).split('/')[-1*label_pos] # -2 is the position of class label varies based on the dataset 
                                                                         #(For ex tiny imagenet has the label position at last but 2 hence it becomes -3)
                if binary_labelling:
                    if class_name in label_of_interest:
                        class_name_modified = common_class_name
                    else:
                        class_name_modified = class_name   
                else:
                    class_name_modified = class_name                                                       
                if any(ext in filename for ext in extensions) and class_name in given_classes : #only consider valid image extensions and the samples from given class
                    file_list.append(os.path.join(root, filename))
                    class_name_list.append(class_name_modified)
        return file_list,class_name_list


    def __len__(self):
        return len(self.img_list)

    
class dataset_from_embeddings(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,embeddings_search,csv_file_search,embeddings_query,csv_file_query,rel_fp,irr_fp,labels,transforms=None,label_pos=-2):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.embeddings_query = embeddings_query
        self.csv_file_query = csv_file_query
        self.embeddings_search = embeddings_search
        self.csv_file_search = csv_file_search
        self.transforms = transforms
        self.label_pos =label_pos

        
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        try:
            db_index = self.csv_file_search.index[self.csv_file_search['file_path'] == img_pth][0]
            emb = self.embeddings_search[db_index]
        except:
            db_index = self.csv_file_query.index[self.csv_file_query['file_path'] == img_pth][0]
            emb = self.embeddings_query[db_index]

        label_idx = self.labels.index(img_pth.split('/')[self.label_pos])
        if self.total_fp[index] in self.irrelevant_fp:
            label = label_idx + 9
        else:
            label = label_idx
        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)


class get_triplet_dataset(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,rel_fp,irr_fp,labels,transforms=None,label_pos=-2):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        label_idx = self.labels.index(img_pth.split('/')[self.label_pos])
        img = Image.open(img_pth)
        if self.total_fp[index] in self.irrelevant_fp:
            label = label_idx + 9
        else:
            label = label_idx
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)

class dataset_from_h5py_csv_dir_3(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,rel_fp,irr_fp,q_label,labels,transforms=None,label_pos=-2,file_mapping_dict_search=None,file_mapping_dict_query=None):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.q_label = q_label
        self.file_mapping_dict_search = file_mapping_dict_search
        self.file_mapping_dict_query = file_mapping_dict_query
        self.path_to_search = []
        for cnt,i in enumerate(self.total_fp):
            try:
                search_path = self.file_mapping_dict_search[i]
            except:
                search_path = self.file_mapping_dict_query[i]

            self.path_to_search.append(search_path)
       
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        csv_search_path = self.path_to_search[index]
        emb_search_path = csv_search_path.replace('.csv','.h5')   
        annot = pd.read_csv(csv_search_path)
        db = h5py.File(emb_search_path, 'r')
        db_embeddings = np.array(db["embed"])
        index_req =  list(annot.file_path).index(img_pth)
        embedding_req = db_embeddings[index_req]
        emb = embedding_req.reshape(1,512,embedding_req.shape[2],embedding_req.shape[2])
        label_from_fp = img_pth.split('/')[self.label_pos]
        if self.q_label=='Tumor':
            if label_from_fp not in ['Benign','InSitu','Invasive','Tumor']:
                label = 0
            if label_from_fp in ['Benign','InSitu','Invasive','Tumor']:
                label = 1
        else:
            if label_from_fp !=self.q_label:
                label = 0
            else:
                label = 1

        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)


class dataset_from_h5py_csv_dir_2(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,rel_fp,irr_fp,q_label,labels,transforms=None,label_pos=-2,file_mapping_dict_search=None,file_mapping_dict_query=None):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.q_label = q_label
        self.file_mapping_dict_search = file_mapping_dict_search
        self.file_mapping_dict_query = file_mapping_dict_query
        t1=time.time()
        print("started collecting emb list")
        for cnt,i in enumerate(self.total_fp):
            try:
                csv_search_path = self.file_mapping_dict_search[i]
            except:
                csv_search_path = self.file_mapping_dict_query[i]

            emb_search_path = csv_search_path.replace('.csv','.h5')
            annot = pd.read_csv(csv_search_path)
            index_req =  list(annot.file_path).index(i)
            db = h5py.File(emb_search_path, 'r')
            db_embeddings = np.array(db["embed"])
            embedding_req = db_embeddings[index_req].reshape(1,db_embeddings.shape[1],db_embeddings.shape[2],db_embeddings.shape[3])
            if cnt !=0:
                self.embeddings = np.concatenate((self.embeddings,embedding_req)) 
            if cnt==0:
                self.embeddings = embedding_req.copy()
        print("Done getting embeddings",time.time()-t1)
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        emb = self.embeddings[index]
        label_from_fp = img_pth.split('/')[self.label_pos]
        if self.q_label=='Tumor':
            if label_from_fp not in ['Benign','InSitu','Invasive','Tumor']:
                label = 0
            if label_from_fp in ['Benign','InSitu','Invasive','Tumor']:
                label = 1
        else:
            if label_from_fp !=self.q_label:
                label = 0
            else:
                label = 1

        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)


class dataset_from_h5py_csv_dir_supervised(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,file_dir1,file_dir2,file_dir3,rel,labels,transforms=None,label_pos=-2):
        rel_copy = copy.deepcopy(rel)
        for cnt,i in enumerate(rel_copy.keys()):
            if cnt ==0:
                self.total_fp = rel_copy[i]
            else:
                self.total_fp.extend(rel_copy[i])
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.emb_files = glob.glob(file_dir1+'/*.h5')
        self.emb_files.sort()
        self.annot_files = glob.glob(file_dir1+'/*.csv')
        self.annot_files.sort()
        self.emb_files_q = glob.glob(file_dir2+'/*.h5')
        self.emb_files_q.sort()
        self.annot_files_q = glob.glob(file_dir2+'/*.csv')
        self.annot_files_q.sort()
        self.emb_files_an = glob.glob(file_dir3+'/*.h5')
        self.emb_files_an.sort()
        self.annot_files_an = glob.glob(file_dir3+'/*.csv')
        self.annot_files_an.sort()
        embeddings_list = []
        meta_data_list = []
        for i in zip(self.emb_files,self.annot_files):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
            req_indices = search_annot.index[search_annot['file_path'].isin(self.total_fp)].tolist()
            req_emb = search_emb[req_indices]
            req_annot = search_annot.iloc[req_indices]
            req_annot = req_annot.reset_index(drop=True)
            embeddings_list.append(req_emb)
            meta_data_list.append(req_annot)
            
        for i in zip(self.emb_files_q,self.annot_files_q):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
            req_indices = search_annot.index[search_annot['file_path'].isin(self.total_fp)].tolist()
            req_emb = search_emb[req_indices]
            req_annot = search_annot.iloc[req_indices]
            req_annot = req_annot.reset_index(drop=True)
            embeddings_list.append(req_emb)
            meta_data_list.append(req_annot)

        for i in zip(self.emb_files_an,self.annot_files_an):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
            req_indices = search_annot.index[search_annot['file_path'].isin(self.total_fp)].tolist()
            req_emb = search_emb[req_indices]
            req_annot = search_annot.iloc[req_indices]
            req_annot = req_annot.reset_index(drop=True)
            embeddings_list.append(req_emb)
            meta_data_list.append(req_annot)
        
        self.csv_file = pd.concat(meta_data_list, ignore_index=True)
        self.emb = np.concatenate(embeddings_list)
        print(len(self.emb),len(self.total_fp))
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        req_index = list(self.csv_file['file_path']).index(img_pth)              
        emb = self.emb[req_index]
        label_from_fp = img_pth.split('/')[self.label_pos]
        label = self.labels.index(label_from_fp)
        # emb = self.transforms(emb)
        
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)



class dataset_from_h5py_csv_dir(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,file_dir1,file_dir2,rel_fp,irr_fp,q_label,labels,transforms=None,label_pos=-2,file_mapping_dict_search=None,file_mapping_dict_query=None):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.emb_files = glob.glob(file_dir1+'/*.h5')
        self.emb_files.sort()
        self.annot_files = glob.glob(file_dir1+'/*.csv')
        self.annot_files.sort()
        self.emb_files_q = glob.glob(file_dir2+'/*.h5')
        self.emb_files_q.sort()
        self.annot_files_q = glob.glob(file_dir2+'/*.csv')
        self.annot_files_q.sort()
        self.q_label = q_label
        embeddings_list = []
        meta_data_list = []
        for i in zip(self.emb_files,self.annot_files):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
            req_indices = search_annot.index[search_annot['file_path'].isin(self.total_fp)].tolist()
            req_emb = search_emb[req_indices]
            req_annot = search_annot.iloc[req_indices]
            req_annot = req_annot.reset_index(drop=True)
            req_annot.loc[req_annot['class_name']== q_label,'class_id'] = 1
            req_annot.loc[req_annot['class_name']!= q_label,'class_id'] = 0
            req_annot.loc[req_annot['class_name']!= q_label,'class_name'] = "Non_"+q_label
            embeddings_list.append(req_emb)
            meta_data_list.append(req_annot)
            
        for i in zip(self.emb_files_q,self.annot_files_q):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
            req_indices = search_annot.index[search_annot['file_path'].isin(self.total_fp)].tolist()
            req_emb = search_emb[req_indices]
            req_annot = search_annot.iloc[req_indices]
            req_annot = req_annot.reset_index(drop=True)
            req_annot.loc[req_annot['class_name']== q_label,'class_id'] = 1
            req_annot.loc[req_annot['class_name']!= q_label,'class_id'] = 0
            req_annot.loc[req_annot['class_name']!= q_label,'class_name'] = "Non_"+q_label
            embeddings_list.append(req_emb)
            meta_data_list.append(req_annot)
        
        self.csv_file = pd.concat(meta_data_list, ignore_index=True)
        self.emb = np.concatenate(embeddings_list)
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        req_index = list(self.csv_file['file_path']).index(img_pth)              
        emb = self.emb[req_index]
        label_from_fp = img_pth.split('/')[self.label_pos]
        if self.q_label=='Tumor':
            if label_from_fp not in ['Benign','InSitu','Invasive','Tumor']:
                label = 0
            if label_from_fp in ['Benign','InSitu','Invasive','Tumor']:
                label = 1
        else:
            if label_from_fp !=self.q_label:
                label = 0
            else:
                label = 1
        if self.transforms:
            emb = self.transforms(emb)
        
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)


class dataset_from_h5py_csv_dir_for_val(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,file_dir,labels,q_label,transforms=None,label_pos=-2,num_samples=None):
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.emb_files = glob.glob(file_dir+'/*.h5')
        self.emb_files.sort()
        self.annot_files = glob.glob(file_dir+'/*.csv')
        self.annot_files.sort()
        self.total_samples = num_samples
        self.q_label = q_label
        self.num_pos_samples = math.ceil(num_samples/2)
        self.num_neg_samples = num_samples - math.ceil(num_samples/2)
        embeddings_list_pos = []
        meta_data_list_pos = []
        embeddings_list_neg = []
        meta_data_list_neg = []
        for i in zip(self.emb_files,self.annot_files):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])
            if self.q_label == 'Tumor':
                search_annot.loc[search_annot['class_name']== 'Normal','class_id'] = 0
                search_annot.loc[search_annot['class_name']!= 'Normal','class_id'] = 1
                search_annot.loc[search_annot['class_name']!= 'Normal','class_name'] = q_label    
            search_annot.loc[search_annot['class_name']== q_label,'class_id'] = 1
            search_annot.loc[search_annot['class_name']!= q_label,'class_id'] = 0
            search_annot.loc[search_annot['class_name']!= q_label,'class_name'] = "Non_"+q_label
            pos_indices = search_annot.index[search_annot['class_name']==q_label].tolist()
            neg_indices = search_annot.index[search_annot['class_name']!=q_label].tolist()
            pos_emb = search_emb[pos_indices]
            neg_emb = search_emb[neg_indices]
            pos_annot = search_annot.iloc[pos_indices]
            pos_annot = pos_annot.reset_index(drop=True)
            neg_annot = search_annot.iloc[neg_indices]
            neg_annot = neg_annot.reset_index(drop=True)
            embeddings_list_pos.append(pos_emb)
            meta_data_list_pos.append(pos_annot)
            embeddings_list_neg.append(neg_emb)
            meta_data_list_neg.append(neg_annot)
        
        csv_file_pos = pd.concat(meta_data_list_pos, ignore_index=True)
        emb_pos = np.concatenate(embeddings_list_pos)
        csv_file_neg = pd.concat(meta_data_list_neg, ignore_index=True)
        emb_neg = np.concatenate(embeddings_list_neg)

        final_pos_indices = random.sample(range(0,len(csv_file_pos)),self.num_pos_samples)
        final_neg_indices = random.sample(range(0,len(csv_file_neg)),self.num_neg_samples)

        pos_annot1 = csv_file_pos.iloc[final_pos_indices]
        pos_annot1 = pos_annot1.reset_index(drop=True)

        neg_annot1 = csv_file_neg.iloc[final_neg_indices]
        neg_annot1 = neg_annot1.reset_index(drop=True)

        self.final_emb = np.concatenate((emb_pos[final_pos_indices],emb_neg[final_neg_indices]))
        self.final_annot = pd.concat([pos_annot1,neg_annot1],ignore_index=True)
        print(len(self.final_annot))
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = list(self.final_annot.file_path)[index]
        emb = self.final_emb[index]
        
        label_from_fp = img_pth.split('/')[self.label_pos]
        if self.q_label=='Tumor':
            if label_from_fp not in ['Benign','InSitu','Invasive','Tumor']:
                label = 0
            if label_from_fp in ['Benign','InSitu','Invasive','Tumor']:
                label = 1
        else:
            if label_from_fp !=self.q_label:
                label = 0
            else:
                label = 1
        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return self.total_samples


class dataset_from_rel_irr_with_transforms(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,rel,irr,transforms=None):
        self.rel = rel
        self.irr = irr
        self.total_fp = rel+irr
        self.transforms = transforms
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        img = Image.open(img_pth)
        if img_pth in self.rel:
            label = 1
        if img_pth in self.irr:
            label = 0
        if self.transforms:
            out_tensor = self.transforms(img)
        return out_tensor,label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)



class dataset_from_img_file_list_full_sup(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,file_dict,labels,transforms=None,label_pos=-2):
        self.total_fp=[]
        dict_copy = copy.deepcopy(file_dict)
        for cnt,i in enumerate(dict_copy.keys()):
            if cnt ==0:
                self.total_fp = dict_copy[i]
            else:
                self.total_fp.extend(dict_copy[i])
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        img = Image.open(img_pth)
        label_from_fp = img_pth.split('/')[self.label_pos]
        label = self.labels.index(label_from_fp)
        if self.transforms:
            out_tensor = self.transforms(img)
        return out_tensor,label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)

class dataset_from_img_file_list(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,img_file_list,q_label,labels,transforms=None,label_pos=-2):
        self.total_fp = img_file_list
        self.transforms = transforms
        self.labels = labels
        self.label_pos = label_pos
        self.q_label = q_label
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        img = Image.open(img_pth)
        label_from_fp = img_pth.split('/')[self.label_pos]
        if self.q_label=='Tumor':
            if label_from_fp not in ['Benign','InSitu','Invasive','Tumor']:
                label = 0
            if label_from_fp in ['Benign','InSitu','Invasive','Tumor']:
                label = 1
        else:
            if label_from_fp !=self.q_label:
                label = 0
            else:
                label = 1

        if self.transforms:
            out_tensor = self.transforms(img)
        return out_tensor,label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)



class dataset_from_embeddings_list(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,search_emb_list,search_csv_file_list, query_emb_list,query_csv_file_list,rel_fp,irr_fp,labels,transforms=None,label_pos=-2):
        self.relevant_fp = rel_fp
        self.irrelevant_fp = irr_fp
        self.total_fp = self.relevant_fp+self.irrelevant_fp
        self.transforms = transforms
        self.labels = labels
        self.search_emb_list = search_emb_list
        self.search_csv_file_list = search_csv_file_list
        self.query_emb_list = query_emb_list
        self.query_csv_file_list = query_csv_file_list
        self.label_pos = label_pos
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = self.total_fp[index]
        for i in range(len(self.search_emb_list)):
            if img_pth in list(self.search_csv_file_list[i].file_path):
                db_index = self.search_csv_file_list[i].index[self.search_csv_file_list[i]['file_path'] == img_pth][0]
                emb = self.search_emb_list[i][db_index]
        
        for i in range(len(self.query_emb_list)):
            if img_pth in list(self.query_csv_file_list[i].file_path):
                db_index = self.query_csv_file_list[i].index[self.query_csv_file_list[i]['file_path'] == img_pth][0]
                emb = self.query_emb_list[i][db_index]
        
        label_idx = self.labels.index(img_pth.split('/')[self.label_pos])
        if self.total_fp[index] in self.irrelevant_fp:
            # label = label_idx + 9
            label = 0
        else:
            # label = label_idx
            label = 1
        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.total_fp)

class dataset_from_embeddings_list_for_val(Dataset):
    """Image Loader for Tiny ImageNet."""
    def __init__(self,search_emb_list,search_csv_file_list, q_label,labels,transforms=None,label_pos=-2):
        self.transforms = transforms
        self.labels = labels
        self.search_emb = np.concatenate(search_emb_list)
        csv_files = []
        for i in search_csv_file_list:
            csv_files.append(i)
        self.search_csv_file = pd.concat(csv_files, ignore_index=True)
        self.label_pos = label_pos
        self.q_label = q_label
    def __getitem__(self, index):
        """Get triplets in dataset."""
        img_pth = list(self.search_csv_file.file_path)[index]
        emb = self.search_emb[index]
        label_from_fp = img_pth.split('/')[self.label_pos]
        label_idx = self.labels.index(label_from_fp)
        if label_from_fp !=self.q_label:
            # label = label_idx + 9
            label = 0
        else:
            # label = label_idx
            label = 1
        if self.transforms:
            emb = self.transforms(emb)
        return emb, label,img_pth,img_pth.split('/')[-1]
    def __len__(self):
        """Get the length of dataset."""
        return len(self.search_emb)


class dataset_from_embeddings_metadata(Dataset):
    def __init__(self,embeddings_search,csv_file_search):
        self.embeddings_search = embeddings_search
        self.csv_file_search = csv_file_search
        
        
    def __getitem__(self, index):
        emb = self.embeddings_search[index]
        class_id = self.csv_file_search.class_id[index]
        file_path = self.csv_file_search.file_path[index]
        class_name = self.csv_file_search.class_name[index]
        return emb, class_id, file_path,class_name
    def __len__(self):
        """Get the length of dataset."""
        return self.embeddings_search.shape[0]