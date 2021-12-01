from custom_datasets import get_triplet_dataset,foldered_dataset,dataset_from_embeddings_metadata,dataset_from_h5py_csv_dir_2,dataset_from_h5py_csv_dir_for_val
from get_emb_ret import get_model_embeddings_from_dataloader,get_model_embeddings_from_dataloader_cpu,get_ranked_images,get_ret_results
from fb_iterations import get_rel_irrl,strip_samples_from_db
from model_update_2 import train_model_triplets,train_classifier
from metrics import calculate_P_at_K, calculate_MAP
from utils import get_micro_macro_values
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import common_functions
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import argparse
import h5py
import os
import itertools
from itertools import combinations 
import multiprocessing
from multiprocessing import Pool, RawArray
import copy
import random
from metric_learn import SCML
import time
import shutil
import glob
import faiss
import torch
import collections
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method
import torch.multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method
import encoding
import random
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from multiprocessing import Pool
import math
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")

from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def extract_embeddings_csv(db_file,db_csv_file):
    db = h5py.File(db_file, 'r')
    db_csv = pd.read_csv(db_csv_file)
    db_embeddings = np.array(db["embed"])
    return db_embeddings,db_csv

def get_entropy(x):
    return (-1*(x+1e-08)*np.log(x+1e-08)).sum(axis=1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


def to_be_reviewed_next_based_on_pred(metadata,indices,gt,prob_preds,q_label,num_samples=20):
    num_n_samples = math.ceil(num_samples/2)
    num_p_samples = num_samples-num_n_samples
    predicted_pos_indices = np.where(np.array(prob_preds) ==q_label)
    predicted_neg_indices = np.where(np.array(prob_preds) !=q_label)
    req_n_indices = predicted_neg_indices[0][:num_n_samples]
    req_p_indices = predicted_pos_indices[0][::-1][:num_p_samples]
    
    to_be_reviewed_next = list(np.array(list(metadata['file_path']))[indices[np.array(list(req_p_indices)+list(req_n_indices))]])
    print("Predicted", np.array(prob_preds)[np.array(list(req_p_indices)+list(req_n_indices))],flush=True)
    print("GT", np.array(gt)[np.array(list(req_p_indices)+list(req_n_indices))],flush=True)
    return to_be_reviewed_next


# def to_be_reviewed_next_based_on_pred(metadata,indices,gt,prob_preds,q_label,num_samples=20):
#     num_n_samples = math.ceil(num_samples/2)
#     num_p_samples = num_samples-num_n_samples
#     predicted_neg_indices = np.where(np.array(prob_preds) !=q_label)
#     req_n_indices = predicted_neg_indices[0][:num_n_samples]
#     req_p_indices = req_n_indices+1
#     req_p_indices = req_p_indices[-1*num_p_samples:]
#     to_be_reviewed_next = list(np.array(list(metadata['file_path']))[indices[np.array(list(req_p_indices)+list(req_n_indices))]])
#     print("Predicted", np.array(prob_preds)[np.array(list(req_p_indices)+list(req_n_indices))],flush=True)
#     print("GT", np.array(gt)[np.array(list(req_p_indices)+list(req_n_indices))],flush=True)
#     return to_be_reviewed_next
    


# def to_be_reviewed_next_based_on_pred(metadata,indices,gt,prob_preds,q_label,num_samples=20):
#     to_be_reviewed_next = []
#     cnt=0
#     for i in range(len(prob_preds)):
#         if prob_preds[i]!=q_label:
#             print("Predicted:{}, GT:{}".format(prob_preds[i],gt[i]))
#             reqd_list = np.array(list(metadata['file_path']))[indices[i:i+2]]
#             cnt+=2
#             to_be_reviewed_next.extend(list(reqd_list))    
#             if cnt>=num_samples:
#                 break
#     return to_be_reviewed_next


def to_be_reviewed_next_based_on_uncertainity(metadata,indices,prob,num_samples=20):
    to_be_reviewed_next = []
    # all_non_match_indices = np.array([i for i in indices if prob_preds[i]!=q_label])
    file_paths_sorted = np.array(list(metadata['file_path']))[indices]
    uncertainity_vector = get_entropy(prob)
    print(uncertainity_vector.shape)
    most_uncertain_indices = np.argsort(uncertainity_vector)[::-1][:num_samples]
    to_be_reviewed_next = list(file_paths_sorted[most_uncertain_indices])
    return to_be_reviewed_next




class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def log_prediction_reports(writer_pred,report,q_label,q_name,fb_round):
    pred_non_q_label = {}
    pred_q_label = {}
    pred_overall = {}
    pred_non_q_label[q_name] = report['Non_{}'.format(q_label)]['f1-score']
    pred_q_label[q_name] = report[q_label]['f1-score']
    pred_overall[q_name] = report['macro avg']['f1-score']
    writer_pred.add_scalars('Non_{}'.format(q_label), pred_non_q_label,fb_round)
    writer_pred.add_scalars(q_label, pred_q_label,fb_round)
    writer_pred.add_scalars('overall', pred_overall,fb_round)
    writer_pred.close()

# def get_emb_from_h5_csv(files,q_label,probs):
#     p=probs
#     metadata_list = []
#     search_emb,search_annot = extract_embeddings_csv(files[0],files[1])    
#     search_annot.loc[search_annot['class_name']== q_label,'class_id'] = 1
#     search_annot.loc[search_annot['class_name']!= q_label,'class_id'] = 0
#     search_annot.loc[search_annot['class_name']!= q_label,'class_name'] = "Non_"+q_label
#     dataset = dataset_from_embeddings_metadata(search_emb,search_annot)
#     dataloader = torch.utils.data.DataLoader(dataset,batch_size=512,shuffle=False,num_workers=0)
#     embeddings,meta_data = get_model_embeddings_from_dataloader(base_model_copy,dataloader,probs=p)
#     meta_data['file_mapping'] = list(np.array([files[1]]*len(meta_data)))
#     return embeddings.reshape(-1,512),list(meta_data.class_id),list(meta_data.class_name),list(meta_data.file_path),list(meta_data.file_mapping)

# def get_emb_metadata(search_dir,q_label,probs=0):
#     p=probs
#     search_emb_files = glob.glob(search_dir+'/*.h5')
#     search_emb_files.sort()
#     search_annot_files = glob.glob(search_dir+'/*.csv')
#     search_annot_files.sort()
#     pool = Pool(multiprocessing.cpu_count())
#     P = itertools.product(zip(search_emb_files,search_annot_files),[q_label],[p])
#     res = pool.starmap(get_emb_from_h5_csv, P)
#     emb_final = np.array(list(res[0][0]))
#     df = {'class_id': list(res[0][1]),'file_path': list(res[0][3]),'class_name': list(res[0][2]),'file_mapping':list(res[0][4])}

#     dataframe = pd.DataFrame(df)
#     print(emb_final.shape,dataframe.shape)
        
#     return emb_final,dataframe
        

def get_emb_metadata(model,search_dir,q_label,probs=0):
    search_emb_files = glob.glob(search_dir+'/*.h5')
    search_emb_files.sort()
    search_annot_files = glob.glob(search_dir+'/*.csv')
    search_annot_files.sort()
    metadata_list = []
    logits= None
    for cnt,i in enumerate(zip(search_emb_files,search_annot_files)):
        search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
        search_annot.loc[search_annot['class_name']== 'Normal','class_id'] = 0
        search_annot.loc[search_annot['class_name']!= 'Normal','class_id'] = 1
        search_annot.loc[search_annot['class_name']!= 'Normal','class_name'] = 'Tumor'
        labels_applicable = ['Invasive','InSitu','Benign','Tumor']
        search_annot.loc[search_annot['class_name']== args.classes_to_query[0],'class_id'] = 1
        search_annot.loc[search_annot['class_name']!= args.classes_to_query[0],'class_id'] = 0
        search_annot.loc[search_annot['class_name']!= args.classes_to_query[0],'class_name'] = "Non_"+args.classes_to_query[0]
        dataset = dataset_from_embeddings_metadata(search_emb,search_annot)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=8192,shuffle=False,num_workers=0)
        embeddings,meta_data = get_model_embeddings_from_dataloader(model,dataloader,probs=probs)
        meta_data['file_mapping'] = list(np.array([i[1]]*len(meta_data)))
        metadata_list.append(meta_data)
        if cnt!=0:
            embeddings_combined = np.concatenate((embeddings_combined,embeddings))
            if logits is not None:
                logits_combined = np.concatenate((logits_combined,logits))
        if cnt==0:
            embeddings_combined = embeddings.copy()
            if logits is not None:
                logits_combined = logits
    meta_data_pd = pd.concat(metadata_list,ignore_index=True)
    if logits is not None:
        return embeddings_combined,logits_combined,meta_data_pd
    else:
        return embeddings_combined,meta_data_pd


def get_pred_results(embeddings_search_original,meta_data_search,reviewed_images_overall,q_label,q_id,trial,csv_init=None,perform_val_only=False,
                     use_knn=False,use_classifier_predictions=False,prob_preds=None,prob_value=None,retrieved_indices=None,num_Neigh=None):
    positives_index = meta_data_search.index[meta_data_search['class_name']==q_label].tolist()
    negatives_index = meta_data_search.index[meta_data_search['class_name']!=q_label].tolist()
    positive_embeddings = embeddings_search_original[positives_index[:500]]
    negative_embeddings = embeddings_search_original[negatives_index[:500]]
    embeddings_to_plot = np.concatenate((positive_embeddings,negative_embeddings),axis=0)
    class_labels_to_plot = [1]*500 + [0]*500
    labels = list(set(meta_data_search.class_name))
    
    if use_classifier_predictions:
        meta_data_search_copy = meta_data_search.copy()
        labels = ['Non_'+q_label,q_label]
        meta_data_search_copy['Final_Pred'] = np.array(['Non_'+q_label]*len(meta_data_search_copy))
        meta_data_search_copy['Confidence'] = np.array([0]*len(meta_data_search_copy))
        meta_data_search_copy['Reviewed'] = np.array([0]*len(meta_data_search_copy))

        reviewed_indices = meta_data_search_copy.index[meta_data_search['file_path'].isin(reviewed_images_overall)].tolist()        

        meta_data_search_copy['Final_Pred'][list(retrieved_indices.reshape(-1))] = prob_preds.reshape(-1)[retrieved_indices.reshape(-1)]
        meta_data_search_copy['Confidence'][list(retrieved_indices.reshape(-1))] = prob_value.reshape(-1)[retrieved_indices.reshape(-1)]
        meta_data_search_copy['Final_Pred'][reviewed_indices] = meta_data_search_copy['class_name'][reviewed_indices]
        meta_data_search_copy['Confidence'][reviewed_indices] = np.array([1]*len(reviewed_indices))
        meta_data_search_copy['Reviewed'][reviewed_indices] = np.array([1]*len(reviewed_indices))
        csv_to_be_saved = meta_data_search_copy.copy()
        
        classes = ['Non_{}'.format(q_label),q_label]
        predictions = np.array(list(meta_data_search_copy['Final_Pred']))
        gt = np.array(list(meta_data_search_copy['class_name']))
        print(set(list(gt)),set(list(predictions)),flush=True)
        report_search = classification_report(gt, predictions, target_names=classes, output_dict=True, digits=4)

        csv_to_be_saved.columns = ['class_id', 'file_path', 'class_name', 'file_mapping','Final_Pred_{}_trial_{}'.format(q_id,trial),'Confidence_{}_trial_{}'.format(q_id,trial),'Reviewed_{}_trial_{}'.format(q_id,trial)]
        if csv_init is not None:
            csv_to_be_saved_final = pd.merge(csv_init, csv_to_be_saved, how="outer", on=["class_id", "file_path","class_name"])
        else:
            csv_to_be_saved_final = csv_to_be_saved
        return csv_to_be_saved_final,report_search
        

    if perform_val_only:
        if len(embeddings_search_original)>1000:
            use_gpu_support=False
        else:
            use_gpu_support=False            
        # writer_embeddings.add_embedding(embeddings_to_plot,metadata=class_labels_to_plot,tag="query_{}_round{}".format(q_id,trial))
        # writer_embeddings.close()
        if num_Neigh is not None:
            N=num_Neigh
        else:
            N = len(embeddings_search_original)
        _,report_P_test,report_AP_test = get_ret_results(embeddings_search_original,embeddings_search_original,meta_data_search,meta_data_search,labels, 
                                                       K_for_P_at_K = min(20,len(embeddings_search_original)),leave_one_out=True,N=N,use_gpu_support=use_gpu_support,
                                                       save_sample_wise_results=True,val_only=True)
        return report_P_test,report_AP_test

    if use_knn:
        reviewed_indices = meta_data_search.index[meta_data_search['file_path'].isin(reviewed_images_overall)].tolist()
        reviewed_emb = embeddings_search_original[reviewed_indices]
        reviewed_annot = meta_data_search.iloc[reviewed_indices]

        not_reviewed_indices = meta_data_search.index[~meta_data_search['file_path'].isin(reviewed_images_overall)].tolist()
        not_reviewed_emb = embeddings_search_original[not_reviewed_indices]
        not_reviewed_annot = meta_data_search.iloc[not_reviewed_indices]
        csv_to_be_saved,report_P,report_AP = get_ret_results(reviewed_emb,not_reviewed_emb,reviewed_annot,not_reviewed_annot,labels, K_for_P_at_K = min(20,len(reviewed_emb)),
                                                             leave_one_out=False,N=len(reviewed_emb),use_gpu_support=False,save_sample_wise_results=True)    
        csv_to_be_saved.columns = ['class_id', 'file_path', 'class_name', 'file_mapping','Final_pred_P@5_{}_trial_{}'.format(q_id,trial),'Final_pred_MAP_{}_trial_{}'.format(q_id,trial)] 
        if csv_init is not None:
            csv_to_be_saved_final = pd.merge(csv_init, csv_to_be_saved, how="outer", on=["class_id", "file_path","class_name"])
        else:
            csv_to_be_saved_final = csv_to_be_saved
        return csv_to_be_saved_final,report_P,report_AP
    
def run_classifier(model,dataset,fb_round,q_name,last_epoch=0,LR=0.0001,perform_validation=0):
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=True,num_workers=0)
    # dataset_val = dataset_from_h5py_csv_dir_for_val(val_dir,labels,q_label,transforms=None,label_pos=-1*args.label_pos,num_samples=int(0.3*len(dataset)))
    dataloader_val = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=True,num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    val_prev_loss = 5000
    for epoch in range(50):
        model,train_loss,val_loss = train_classifier(model, epoch ,device,dataloader, dataloader_val,optimizer)
        losses_classifier_dict_train[q_name] = train_loss
        losses_classifier_dict_val[q_name] = val_loss
        losses_classifier.add_scalars('classifier_loss_train', losses_classifier_dict_train,last_epoch+epoch)
        losses_classifier.add_scalars('classifier_loss_val', losses_classifier_dict_val,last_epoch+epoch)
        # if val_loss<=val_prev_loss:
        #     val_prev_loss = val_loss
        #     model_prev = copy.deepcopy(model)
        # else:
        #     break
    # model_best = copy.deepcopy(model_prev)
    losses_classifier.close()
    return model,last_epoch+epoch
def update_model_metric_learning(model,dataset,fb_round,q_name,last_epoch=0,LR=0.0001):
    num_hard_batch = 0
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=True,num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=True,num_workers=0)
    loss_func = losses.TripletMarginLoss(margin=1.0)
    mining_func = miners.TripletMarginMiner(margin=1.0,type_of_triplets="all")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    val_prev_loss = 50000
    for epoch in range(50):
        model,train_loss,val_loss = train_model_triplets(model, epoch , loss_func, mining_func, device, dataloader, dataloader_val , optimizer)  
        losses_metric_learning_dict_train[q_name] = train_loss
        losses_metric_learning_dict_val[q_name] = val_loss
        losses_metric_learning.add_scalars('Metric_learning_loss_train', losses_metric_learning_dict_train,last_epoch+epoch)
        losses_metric_learning.add_scalars('Metric_learning_loss_val', losses_metric_learning_dict_val,last_epoch+epoch)
        # if val_loss<=val_prev_loss:
        #     val_prev_loss = val_loss
        #     model_prev = copy.deepcopy(model)
        # else:
        #     break
    # model_best = copy.deepcopy(model_prev)
    losses_metric_learning.close()
    return model,last_epoch+epoch




def get_updated_embeddings(q_emb,rel,irr,meta_data_search,meta_data_query,embeddings_search,embeddings_query):
    rel_fp_from_search_db = [ i for i in list(set(rel))]
    irr_fp = [ i for i in list(set(irr))]
    indices_rel_search_db = meta_data_search.index[meta_data_search['file_path'].isin(rel_fp_from_search_db)].tolist()
    indices_irrel = meta_data_search.index[meta_data_search['file_path'].isin(irr_fp)].tolist()
    rel_emb_from_search_db = embeddings_search[indices_rel_search_db]
    irr_emb = embeddings_search[indices_irrel]
    q_emb = q_emb.reshape(1,-1)
    # updated_embeddings = np.mean(rel_emb_from_search_db,axis=0).reshape(1,-1)
    # updated_embeddings = q_emb+np.mean(np.concatenate((rel_emb_from_search_db)),axis=0)-np.mean(np.concatenate((irr_emb)),axis=0)
    updated_embeddings = np.mean(np.concatenate((q_emb,rel_emb_from_search_db)),axis=0)
    return updated_embeddings

def rel_fb_loop(model,classifier,query_idx,embeddings_search_512,meta_data_search,embeddings_query_512,meta_data_query,labels_applicable,rel_images_overall,
                irr_images_overall,reviewed_images_overall,relevant_top_5_pred_based,k=50,use_aux_query=0,csv_res_init=None,
                csv_updated_res_init=None,csv_prob = None,predictions_to_save_csv=None,val_preds_to_save=None,last_epoch_metric_learning = 0,last_epoch_classifier = 0,
                LR=0.0001,perform_validation=0,use_top_5_pred=0): 
    model_copy = copy.deepcopy(model)
    classifier_copy = copy.deepcopy(classifier)

    fb_round = 0
    l = 0

    embeddings_search = embeddings_search_512
    embeddings_query = embeddings_query_512
    
    embeddings_search_original = embeddings_search.copy()
    embeddings_query_original = embeddings_query.copy()
    
    query_image_name = list(meta_data_query.file_path)[query_idx].split('/')[-1]
    query_fp = meta_data_query.file_path[query_idx]
    q_label = meta_data_query.class_name[query_idx]
    q_name = query_fp.split('/')[-2]+query_image_name

    
    
    rel_images_overall.extend([query_fp])
    final_dict = {}
    if use_aux_query:
        aux_query_fp = list(set(list(meta_data_query.file_path))-set(query_fp))
        aux_query_fp = random.sample(aux_query_fp,1)
        rel_images_overall.extend(aux_query_fp)
    
    print("query image name",query_image_name,flush=True)
    query_emb_to_be_searched = embeddings_query[query_idx].copy()

    indices,preds = get_ranked_images(embeddings_search_original, query_emb_to_be_searched,meta_data_search,leave_one_out=False)
    result = (preds == meta_data_query.class_name[query_idx])*1
    
    final_dict = {}
    final_dict['Query_name'] = [q_name]
    
    
    

    
    # report_P,report_AP = get_pred_results(embeddings_search_original,meta_data_search,reviewed_images_overall,q_label,query_idx,0,perform_val_only=True,num_Neigh=1000)
    # print("initial train reports", report_P)
    # log_prediction_reports(knn_preds,report_P,q_label,q_name,0)
    # log_prediction_reports(classifier_preds,report_P,q_label,q_name,0)
    # log_prediction_reports(knn_preds_res,report_P,q_label,q_name,0)    
    
    while fb_round<5:
        model = copy.deepcopy(model_copy)
        model = model.to(device)
        classifier = copy.deepcopy(classifier_copy)
        classifier = classifier.to(device)
        if fb_round==0:
            relevant,irrelevant = get_rel_irrl(reviewed_images_overall,embeddings_search,meta_data_search,query_emb_to_be_searched,q_label,NN=k,strategy = args.ret_strategy,label_pos=args.label_pos)
        else:

            if args.entropy_based:
                relevant = []
                irrelevant = []
                print("Using Entropy based")
                reviewed_images_overall_copy = reviewed_images_overall+relevant+irrelevant
                reviewed_indices = meta_data_search.copy().index[meta_data_search.copy()['file_path'].isin(reviewed_images_overall_copy)].tolist()
                indices_prev = indices.copy().reshape(-1)
                # indices = np.setdiff1d(indices_prev,reviewed_indices)
                indices = np.array([x for x in list(indices_prev) if x not in list(reviewed_indices)])
                prob_values_sorted_search = probs_search[indices.reshape(-1)]
                to_be_reviewed_next = to_be_reviewed_next_based_on_uncertainity(meta_data_search.copy(),indices,prob_values_sorted_search,num_samples=k)
                relevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos] == q_label ]
                irrelevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos] != q_label]
                relevant.extend(relevant_prob_based)
                irrelevant.extend(irrelevant_prob_based)

            if args.classifier_pred_top_mid_end:
                relevant,irrelevant = get_rel_irrl(reviewed_images_overall,embeddings_search,meta_data_search,query_emb_to_be_searched,q_label,NN=int(0.75*k),strategy = args.ret_strategy,label_pos=args.label_pos)
                reviewed_images_overall_copy = reviewed_images_overall+relevant+irrelevant
                reviewed_indices = meta_data_search.copy().index[meta_data_search.copy()['file_path'].isin(reviewed_images_overall_copy)].tolist()
                indices_prev = indices.copy().reshape(-1)
                # indices = np.setdiff1d(indices_prev,reviewed_indices)
                indices = np.array([x for x in list(indices_prev) if x not in list(reviewed_indices)])
                prob_preds_sorted_search = prob_preds_search.reshape(-1)[indices.reshape(-1)]
                gt = np.array(list(meta_data_search['class_name']))[indices]
                to_be_reviewed_next = to_be_reviewed_next_based_on_pred(meta_data_search.copy(),indices,gt,prob_preds_sorted_search,q_label,num_samples=k-int(0.75*k))
                relevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos]in labels_applicable]
                irrelevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos] not in labels_applicable]
                relevant.extend(relevant_prob_based)
                irrelevant.extend(irrelevant_prob_based)

            if args.only_classifier_pred:
                relevant = []
                irrelevant = []
                reviewed_images_overall_copy = reviewed_images_overall+relevant+irrelevant
                reviewed_indices = meta_data_search.index[meta_data_search['file_path'].isin(reviewed_images_overall_copy)].tolist()
                indices_prev = indices.copy().reshape(-1)
                # indices = np.setdiff1d(indices,reviewed_indices)
                indices = np.array([x for x in list(indices_prev) if x not in list(reviewed_indices)])
                prob_preds_sorted_search = prob_preds_search.reshape(-1)[indices.reshape(-1)]
                gt = np.array(list(meta_data_search['class_name']))[indices]
                to_be_reviewed_next = to_be_reviewed_next_based_on_pred(meta_data_search,indices,gt,prob_preds_sorted_search,q_label,num_samples=k)
                relevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos] in labels_applicable]
                irrelevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos] not in labels_applicable]
                relevant.extend(relevant_prob_based)
                irrelevant.extend(irrelevant_prob_based)

            else:
                relevant,irrelevant = get_rel_irrl(reviewed_images_overall,embeddings_search,meta_data_search,query_emb_to_be_searched,q_label,NN=k,strategy = args.ret_strategy,label_pos=args.label_pos)



        
        rel_images_overall.extend((relevant))
        irr_images_overall.extend((irrelevant))
        reviewed_images_overall.extend((relevant+irrelevant))
        

        rel_images_overall = list(set(rel_images_overall))
        irr_images_overall = list(set(irr_images_overall))
        reviewed_images_overall = list(set(reviewed_images_overall))
        print("Rel set class", set([rel_iter.split('/')[-1*args.label_pos] for rel_iter in rel_images_overall]),flush=True)
        print("Irrel set class", set([irrel_iter.split('/')[-1*args.label_pos] for irrel_iter in irr_images_overall]),flush=True)
        print("FB round {} , Rel {} , Top-5 pred based {}, Irr {}, Reviewed in session so far.. {} ".format(fb_round,len(rel_images_overall),len(relevant_top_5_pred_based),len(irr_images_overall),len(reviewed_images_overall)),flush=True)
        t1  =time.time()
        dataset = dataset_from_h5py_csv_dir_2(list(set(rel_images_overall)),list(set(irr_images_overall)),q_label,args.label_set,transforms=None,label_pos=-1*args.label_pos,file_mapping_dict_search=args.file_mapping_dict_search,file_mapping_dict_query=args.file_mapping_dict_query)
    
        model_updated,last_epoch_metric_learning = update_model_metric_learning (model,dataset,fb_round,q_name,last_epoch=last_epoch_metric_learning,LR=LR)
        classifier_updated,last_epoch_classifier = run_classifier(classifier,dataset,fb_round,q_name,last_epoch=last_epoch_classifier,LR=LR,perform_validation=perform_validation)
        print('Metric learning and classifier updation done in ' , time.time()-t1,flush=True)
        
        embeddings_query,meta_data_query = get_emb_metadata(model_updated,args.query_dir,q_label)
        embeddings_search,meta_data_search = get_emb_metadata(model_updated,args.search_dir,q_label)
        
        query_emb_to_be_searched = embeddings_query[query_idx].copy()

        if args.use_query_refinement:
            query_emb_to_be_searched = get_updated_embeddings(query_emb_to_be_searched,rel_images_overall,irr_images_overall,meta_data_search,meta_data_query,embeddings_search,embeddings_query)    
        
        indices,preds = get_ranked_images(embeddings_search, query_emb_to_be_searched.reshape(-1,1),meta_data_search,leave_one_out=False)
        

        # if use_top_5_pred:
        #     ranked_img_fp = list(np.array(list(meta_data_search.file_path))[indices.reshape(-1)])
        #     reqd_relevant_img_fp = list(set(ranked_img_fp[:5+len(rel_images_overall)+len(relevant_top_5_pred_based)]) - set(rel_images_overall+relevant_top_5_pred_based))[:max(0,len(irr_images_overall)-len(rel_images_overall)-len(relevant_top_5_pred_based))]
        #     reqd_relevant_img_fp_rel_only = [i for i in reqd_relevant_img_fp if i.split('/')[-2] in args.classes_to_query[0]]
        #     print('rel images overall -irr images overall,reqd_relevant images',len(irr_images_overall)-len(rel_images_overall),len(reqd_relevant_img_fp_rel_only))
        #     relevant_top_5_pred_based.extend((reqd_relevant_img_fp_rel_only))
        #     relevant_top_5_pred_based = list(set(relevant_top_5_pred_based))
        #     print(np.unique(np.array([i.split('/')[-2] for i in relevant_top_5_pred_based]),return_counts=True))


        probs_query,meta_data_query = get_emb_metadata(classifier_updated,args.query_dir,q_label,probs=1)
        probs_search, meta_data_search  = get_emb_metadata(classifier_updated,args.search_dir,q_label,probs=1)
        
        prob_preds_search = np.array(['Non_{}'.format(q_label),q_label])[np.argmax(probs_search,axis=1)].reshape(-1)
        fb_round+=1
        
       
    csv_prob,report_search = get_pred_results(embeddings_search,meta_data_search,reviewed_images_overall,q_label,query_idx,fb_round,
                                              csv_init=csv_prob,perform_val_only=False,use_knn=False,use_classifier_predictions=True,prob_preds=prob_preds_search,
                                              prob_value=probs_search,retrieved_indices=indices)
    
    # print("classifier report",report_search)
    csv_prob.to_csv(os.path.join(args.save_prediction_csv,'{}_prob_pred.csv'.format(args.classes_to_query[0])))
    log_prediction_reports(classifier_preds,report_search,q_label,q_name,fb_round)


    csv_res_init,report_P,report_AP = get_pred_results(embeddings_search_original,meta_data_search,reviewed_images_overall,q_label,query_idx,fb_round,
                                                      csv_init=csv_res_init,use_knn=True)
    
    # print("KNN report original resnet",report_P)
    csv_res_init.to_csv(os.path.join(args.save_prediction_csv,'{}_res_pred.csv'.format(args.classes_to_query[0])))
    log_prediction_reports(knn_preds_res,report_P,q_label,q_name,fb_round)
    csv_updated_res_init,report_P,report_AP = get_pred_results(embeddings_search,meta_data_search,reviewed_images_overall,q_label,query_idx,fb_round,
                                                             csv_init=csv_updated_res_init,use_knn=True)
    
    # print("KNN report updated resnet",report_P)
    csv_updated_res_init.to_csv(os.path.join(args.save_prediction_csv,'{}_updated_res_pred.csv'.format(args.classes_to_query[0])))
    log_prediction_reports(knn_preds,report_P,q_label,q_name,fb_round)
    
    final_dict['F1_AP@5_based_Step_{}'.format(fb_round)] = [report_AP['macro avg']['f1-score']] 
    final_dict['F1_P@5_based_Step_{}'.format(fb_round)] = [report_P['macro avg']['f1-score']]
    final_dict['F1_classifier_based_Step_{}'.format(fb_round)] =  [report_search['macro avg']['f1-score']]
    final_dict['Accuracy_classifier_based_Step_{}'.format(fb_round)] =  [report_search['accuracy']]

        
        

    df = pd.DataFrame(final_dict)    
    if predictions_to_save_csv is not None:
        predictions_to_save_csv = predictions_to_save_csv.append(df.copy(), ignore_index=True)
        
    else:    
        predictions_to_save_csv = df.copy()
        
    predictions_to_save_csv.to_csv(os.path.join(args.save_prediction_csv,"final_overall_pred_results.csv"))
    
    return rel_images_overall,irr_images_overall,reviewed_images_overall,relevant_top_5_pred_based,csv_res_init,csv_updated_res_init,csv_prob,predictions_to_save_csv,val_preds_to_save,\
           model_updated,classifier_updated,last_epoch_metric_learning,last_epoch_classifier



def get_base_model_trainable_model(resnet_model,layer=4,sub_layer=0):
    if sub_layer==0:
        trainable = nn.Sequential(*list(resnet_model.children())[-((6-layer+1)):],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)
        base = nn.Sequential(*list(resnet_model.children())[:-((6-layer+1))])    
    if sub_layer==1:
        trainable = nn.Sequential(getattr(resnet_model, 'layer{}'.format(layer))[sub_layer],*list(resnet_model.children())[-((6-layer)):],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)
        base = nn.Sequential(*list(resnet_model.children())[:-(6-layer+1)],getattr(resnet_model, 'layer{}'.format(layer))[sub_layer-1])
    return base,trainable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for Running Fb Retrieval from a search DB and a query DB")
    parser.add_argument("--save_results_name", type=str, default = "resenet18", help="method for metric learning")
    parser.add_argument("--search_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--make_query_test_set", type=int, default=0, help="Class Names")
    parser.add_argument("--query_dir", type=str, default='./', help="Class Names")
    parser.add_argument("--num_img_to_rev", type=int, default = 50, help="method for metric learning")
    parser.add_argument("--use_aux_query", type=int, default = 0, help="method for metric learning")
    parser.add_argument("--use_classifier_pred", type=int, default = 0, help="method for metric learning")
    parser.add_argument("--classes_to_query", type=str, nargs='+', required=True, help="Class Names")
    parser.add_argument("--use_query_refinement", type=int, default=0)
    parser.add_argument("--ret_strategy", type=str, default="top_k_ret")
    parser.add_argument("--model", type=str, default="resnet-18")
    parser.add_argument("--save_prediction_csv", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="/ssd_scratch/cvit/ashishmenon/tensorboard_logs_ICIAR4/")
    parser.add_argument("--label_pos", type=int, default=2)
    parser.add_argument("--only_classifier_pred", type=int, default=0)
    parser.add_argument("--classifier_pred_top_mid_end", type=int, default=0)
    parser.add_argument("--entropy_based", type=int, default=0)
    parser.add_argument("--label_set", type=str, nargs='+', default=None, help="Class Names")
    parser.add_argument("--model_save_path", type=str, default='./', help="Class Names")

    
    
    args = parser.parse_args()
    print(args,flush=True)
    os.makedirs(args.save_prediction_csv,exist_ok=True)
    os.makedirs(args.model_save_path,exist_ok=True)
    
    search_emb_files = glob.glob(args.search_dir+'/*.h5')
    search_emb_files.sort()
    search_annot_files = glob.glob(args.search_dir+'/*.csv')
    search_annot_files.sort()

    
    
    q_size = 0
    t_size = 0
    v_size = 0
    query_done = 0
    test_size = 0


    log_file = args.log_dir
    labels_applicable = ['Invasive','InSitu','Benign','Tumor']
    if args.label_set is not None:
        labels_database = args.label_set
    else:
        labels_database = os.listdir(args.dataroot)       

    if args.make_query_test_set:
        print("Making query set")
        mod_train_path = args.search_dir + '_modified/train/'
        mod_query_path = args.search_dir + '_modified/query/'
        
        os.makedirs(mod_train_path,exist_ok=True)
        os.makedirs(mod_query_path,exist_ok=True)
        
        for cnt,i in enumerate(zip(search_emb_files,search_annot_files)):
            search_emb,search_annot = extract_embeddings_csv(i[0],i[1])
            req_paths = [i for i in list(search_annot.file_path) if i.split('/')[-1*args.label_pos] in labels_applicable]
                
            if len(req_paths)!= 0 and q_size!=10:
                query_filepaths = req_paths[:10-q_size]
                if len(query_filepaths)!=0:
                    print("Found query")
                    query_indices = search_annot.index[search_annot['file_path'].isin(query_filepaths)].tolist()
                    query_emb = search_emb[query_indices]
                    query_annot = search_annot.iloc[query_indices]               
                    query_annot = query_annot.reset_index(drop=True)
                    with h5py.File(mod_query_path + "query_{}_emb.h5".format(cnt), 'w') as f:
                        f.create_dataset('embed', data=query_emb)
                    query_annot.to_csv(mod_query_path + '/query_{}_emb.csv'.format(cnt),index=False)
                    q_size+=len(query_emb)
            else:
                query_filepaths = []
                
            query_fp = list(set(query_filepaths))
            train_indices = search_annot.index[~search_annot['file_path'].isin(query_fp)].tolist()
            train_emb = search_emb[train_indices]
            train_annot = search_annot.iloc[train_indices]
            train_annot = train_annot.reset_index(drop=True)
            
            with h5py.File(mod_train_path + "train_{}_emb.h5".format(cnt), 'w') as f:
                    f.create_dataset('embed', data=train_emb)
            train_annot.to_csv(mod_train_path + '/train_{}_emb.csv'.format(cnt),index=False)

        args.search_dir = mod_train_path
        args.query_dir = mod_query_path
        
    
    else:
        c=1

    if args.use_resnet18:
        resnet_model = models.resnet18(pretrained = True)
        resnet_model.fc = Identity()
        trainable_model = nn.Sequential(resnet_model.layer4[1],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)

        resnet_model_classifier = models.resnet18(pretrained = True)
        resnet_model_classifier.fc = nn.Linear(512, 2)
        classifier_trainable_model  = nn.Sequential(resnet_model_classifier.layer4[1],resnet_model_classifier.avgpool,nn.Flatten(),resnet_model_classifier.fc)

    if args.use_texture_encoder:
        base_model_main = nn.Sequential(nn.Conv2d(512,128,(1,1)),nn.BatchNorm2d(128),nn.ReLU(),
                                         encoding.nn.Encoding(128,32),encoding.nn.View(-1,4096),encoding.nn.Normalize(),
                                         nn.Linear(4096,128) )
        classifier_main = nn.Sequential(nn.Conv2d(512,128,(1,1)),nn.BatchNorm2d(128),nn.ReLU(),
                                         encoding.nn.Encoding(128,32),encoding.nn.View(-1,4096),encoding.nn.Normalize(),
                                         nn.Linear(4096,2))


    losses_classifier_dict_train = {}
    losses_classifier_dict_val = {}
    losses_metric_learning_dict_train = {}
    losses_metric_learning_dict_val = {}
    

    rel = []
    irr = []
    rev = []
    csv_res = None
    csv_updated_res = None
    csv_prob = None
    predictions_to_save_csv  = None
    val_preds_to_save = None
    MAP_query_wise_csv = None 
    Pr_20_query_wise_csv = None
    Pr_25P_query_wise_csv = None
    Pr_50P_query_wise_csv = None
    Pr_75P_query_wise_csv = None
    Pr_100P_query_wise_csv = None
    writer_retrieval = SummaryWriter(log_dir=os.path.join(log_file,'Retrieval'))
    writer_embeddings = SummaryWriter(log_dir=os.path.join(log_file,'embeddings'))
    losses_metric_learning = SummaryWriter(log_dir=os.path.join(log_file,'losses_metric_learning'))
    
    losses_classifier = SummaryWriter(log_dir=os.path.join(log_file,'losses_classifier'))

    classifier_preds = SummaryWriter(log_dir=os.path.join(log_file,'classifer_F1_results'))
    knn_preds = SummaryWriter(log_dir=os.path.join(log_file,'knn_F1_results_updated_embeddings'))
    knn_preds_res = SummaryWriter(log_dir=os.path.join(log_file,'knn_F1_results_resnet18'))

    
    trainable_model_copy = copy.deepcopy(trainable_model)
    classifier_trainable_model_copy = copy.deepcopy(classifier_trainable_model)
    embeddings_query_512,meta_data_query = get_emb_metadata(trainable_model_copy.to(device),args.query_dir,args.classes_to_query[0])
        
    embeddings_search_512 , meta_data_search  = get_emb_metadata(trainable_model_copy.to(device),args.search_dir,args.classes_to_query[0])

    
    
    args.file_mapping_dict_search = dict(zip(meta_data_search.file_path, meta_data_search.file_mapping))
    args.file_mapping_dict_query = dict(zip(meta_data_query.file_path, meta_data_query.file_mapping))
    # args.file_mapping_dict_val = dict(zip(meta_data_val.file_path, meta_data_val.file_mapping))
    
    print("Loaded embeddings and annot done with search query ",len(embeddings_search_512),len(embeddings_query_512),flush=True)

    num_img_to_rev = args.num_img_to_rev
    last_epoch_metric_learning = 0 
    last_epoch_classifier = 0
    LR = 0.0001
    labels_applicable = ['Invasive','InSitu','Benign','Tumor']
    perform_validation = 0
    relevant_top_5_pred_based = []
    use_top_5_pred = 0
    for i in list(meta_data_query.file_path[:10]):
        index_req = list(meta_data_query.file_path).index(i)
        base_model = copy.deepcopy(trainable_model)
        classifier = copy.deepcopy(classifier_trainable_model)
        print("Starting with query image {}".format(i),flush=True)
        tq = time.time()
        rel,irr,rev,relevant_top_5_pred_based,csv_res,csv_updated_res,csv_prob,predictions_to_save_csv,val_preds_to_save,m1,m2,last_epoch_metric_learning,last_epoch_classifier =  \
                                                                               rel_fb_loop(base_model.to(device),classifier.to(device),index_req,
                                                                               embeddings_search_512,meta_data_search, 
                                                                               embeddings_query_512, meta_data_query,
                                                                               labels_applicable,rel,irr,rev,relevant_top_5_pred_based,k=num_img_to_rev,use_aux_query=args.use_aux_query,
                                                                               csv_res_init=csv_res,csv_updated_res_init=csv_updated_res,csv_prob=csv_prob,
                                                                               predictions_to_save_csv=predictions_to_save_csv,val_preds_to_save=val_preds_to_save,
                                                                               last_epoch_metric_learning=last_epoch_metric_learning,last_epoch_classifier=last_epoch_classifier,LR=LR,perform_validation=perform_validation,use_top_5_pred=use_top_5_pred)
        
        query_image_name = list(np.array(meta_data_query.file_path))[index_req].split('/')[-1].split('.')[0]
        query_image_label = list(np.array(meta_data_query.file_path))[index_req].split('/')[-1*args.label_pos].split('.')[0]
        query_slide_name = list(np.array(meta_data_query.file_path))[index_req].split('/')[-3].split('.')[0]
        print('Time taken for query image {} is {}'.format(i,time.time()-tq),flush=True)        
        losses_classifier_dict_train = {}
        losses_classifier_dict_val = {}
        losses_metric_learning_dict_train = {}
        losses_metric_learning_dict_val = {}
        torch.save(m1.state_dict(), "{}/metric_learning_model_{}_{}_{}.pth".format(args.model_save_path,query_image_name,query_slide_name,query_image_label))
        torch.save(m2.state_dict(), "{}/classifier_model_{}_{}_{}.pth".format(args.model_save_path,query_image_name,query_slide_name,query_image_label))
        perform_validation=1
        use_top_5_pred = 1
           
        
