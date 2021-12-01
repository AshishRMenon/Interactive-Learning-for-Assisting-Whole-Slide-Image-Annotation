from custom_datasets import dataset_from_embeddings_metadata,dataset_from_h5py_csv_dir,dataset_from_h5py_csv_dir_supervised
from get_emb_ret import get_model_embeddings_from_dataloader,get_ranked_images,get_ret_results_val
from fb_iterations import get_rel_irrl
from model_update_2 import train_model_triplets,train_classifier
from metrics import calculate_P_at_K, calculate_MAP
from utils import get_micro_macro_values
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import common_functions
from sklearn.metrics import classification_report
import numpy as np
import math
import pandas as pd
import argparse
import h5py
import os
import copy
import random
import time
import shutil
import glob
import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")

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
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def to_be_reviewed_next_based_on_pred(metadata,indices,gt,prob_preds,q_label,num_samples=20):
    num_n_samples = math.ceil(num_samples/2)
    num_p_samples = num_samples-num_n_samples
    predicted_pos_indices = np.where(np.array(prob_preds) ==q_label)
    predicted_neg_indices = np.where(np.array(prob_preds) !=q_label)
    req_n_indices = predicted_neg_indices[0][:num_n_samples]
    req_p_indices = predicted_pos_indices[0][::-1][:num_p_samples]
    to_be_reviewed_next = list(np.array(list(metadata['file_path']))[indices[np.array(list(req_p_indices)+list(req_n_indices))]])
    return to_be_reviewed_next



def to_be_reviewed_next_based_on_uncertainity(metadata,indices,prob,num_samples=20):
    to_be_reviewed_next = []
    # all_non_match_indices = np.array([i for i in indices if prob_preds[i]!=q_label])
    file_paths_sorted = np.array(list(metadata['file_path']))[indices]
    uncertainity_vector = get_entropy(prob)
    most_uncertain_indices = np.argsort(uncertainity_vector)[::-1][:num_samples]
    to_be_reviewed_next = list(file_paths_sorted[most_uncertain_indices])
    return to_be_reviewed_next




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


def get_emb_metadata_from_h5py_csv(model,search_dir,probs=0):
    search_emb_files = glob.glob(search_dir+'/*.h5')
    search_emb_files.sort()
    search_annot_files = glob.glob(search_dir+'/*.csv')
    search_annot_files.sort()
    metadata_list = []
    logits= None
    for cnt,i in enumerate(zip(search_emb_files,search_annot_files)):
        search_emb,search_annot = extract_embeddings_csv(i[0],i[1])    
        dataset = dataset_from_embeddings_metadata(search_emb,search_annot)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=8192,shuffle=False,num_workers=0)
        embeddings,meta_data = get_model_embeddings_from_dataloader(model,dataloader,probs=probs)
        metadata_list.append(meta_data)
        if cnt!=0:
            embeddings_combined = np.concatenate((embeddings_combined,embeddings))
            if logits is not None:
                logits_combined = np.concatenate((logits_combined,logits))
        if cnt==0:
            embeddings_combined = embeddings
            if logits is not None:
                logits_combined = logits
    meta_data_pd = pd.concat(metadata_list,ignore_index=True)
    if logits is not None:
        return embeddings_combined,logits_combined,meta_data_pd
    else:
        return embeddings_combined,meta_data_pd

    
def run_classifier(model,search_dir,query_dir,annotated_dir,relevant_dict,labels,fb_round,q_name,last_epoch=0):
    dataset = dataset_from_h5py_csv_dir_supervised(search_dir,query_dir,annotated_dir,relevant_dict,labels,transforms=None,label_pos=-1*args.label_pos)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=True,num_workers=0)
    
    dataloader_val = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),512),shuffle=False,num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    val_prev_loss = 5000
    for epoch in range(50):
        model,train_loss,val_loss = train_classifier(model, epoch ,device,dataloader, dataloader_val,optimizer)
        losses_classifier_dict_train[q_name] = train_loss
        losses_classifier_dict_val[q_name] = val_loss
        losses_classifier.add_scalars('classifier_loss_train', losses_classifier_dict_train,last_epoch+epoch)
        losses_classifier.add_scalars('classifier_loss_val', losses_classifier_dict_val,last_epoch+epoch)
    losses_classifier.close()
    return model,last_epoch+epoch



def update_model_metric_learning(model,search_dir, query_dir, annotated_dir,relevant_dict,labels,fb_round,q_name,last_epoch=0):
    num_hard_batch = 0
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
                           ])
    
    dataset = dataset_from_h5py_csv_dir_supervised(search_dir,query_dir,annotated_dir,relevant_dict,labels,transforms=None,label_pos=-1*args.label_pos)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),256),shuffle=True,num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset,batch_size=min(len(dataset),256),shuffle=False,num_workers=0)
    
    
    loss_func = losses.TripletMarginLoss(margin=0.2)
    mining_func = miners.TripletMarginMiner(margin=0.2,type_of_triplets="hard")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    val_prev_loss = 5000
    for epoch in range(50):
        model,train_loss,val_loss = train_model_triplets(model, epoch , loss_func, mining_func, device, dataloader, dataloader_val , optimizer,perform_val=1)  
        losses_metric_learning_dict_train[q_name] = train_loss
        losses_metric_learning_dict_val[q_name] = val_loss
        losses_metric_learning.add_scalars('Metric_learning_loss_train', losses_metric_learning_dict_train,last_epoch+epoch)
        losses_metric_learning.add_scalars('Metric_learning_loss_val', losses_metric_learning_dict_val,last_epoch+epoch)
    losses_metric_learning.close()
    return model,last_epoch+epoch



def get_updated_embeddings(q_emb,rel,meta_data_search,meta_data_annotated,meta_data_query,embeddings_search,embeddings_annot,embeddings_query):
    rel_fp = [ i for i in list(set(rel))]
    indices_rel_search_db = meta_data_search.index[meta_data_search['file_path'].isin(rel_fp)].tolist()
    indices_rel_annotated_db = meta_data_annotated.index[meta_data_annotated['file_path'].isin(rel_fp)].tolist()
    rel_emb_from_search_db = embeddings_search[indices_rel_search_db]
    rel_emb_from_annotated_db = embeddings_annot[indices_rel_annotated_db]
    q_emb = q_emb.reshape(1,-1)
    updated_embeddings = np.mean(np.concatenate((q_emb,rel_emb_from_search_db,rel_emb_from_annotated_db)),axis=0)
    return updated_embeddings

def rel_fb_loop(model,classifier,embeddings_search_512,embeddings_query_512,embeddings_val_512,
                embeddings_test_512, meta_data_search,meta_data_query, meta_data_val, 
                meta_data_test,rel_images_dict,k=50,use_aux_query=0): 
    model_copy = copy.deepcopy(model)
    classifier_copy = copy.deepcopy(classifier)


    embeddings_query  = embeddings_query_512.copy()
        
    embeddings_search = embeddings_search_512.copy()

    embeddings_annotated = embeddings_annotated_512.copy()

    embeddings_test  = embeddings_test_512.copy()

    embeddings_search_original = embeddings_search.copy()
    embeddings_query_original = embeddings_query.copy()
    embeddings_annotated_original = embeddings_annotated.copy()
    embeddings_test_original = embeddings_test.copy()
    


    reviewed_images_overall  = []
    reqd_q_indices_list = [0,10,20,30,40,50,60,70,80]
    for r in range(1):
        for q in reqd_q_indices_list:
            query_idx = q+r
            print("Starting with query image {}".format(query_idx),flush=True)    
            query_image_name = list(np.array(meta_data_query.file_path))[query_idx].split('/')[-1]
            query_fp = meta_data_query.file_path[query_idx]
            q_label = meta_data_query.class_name[query_idx]
            q_name = query_fp.split('/')[-2]+query_image_name

            print("file path name and label",query_fp,q_name,q_label,flush=True)
            rel_images_dict[q_label].extend([query_fp])
            
            if use_aux_query:
                aux_query_fp = list(set(list(meta_data_query.file_path))-set(query_fp))
                aux_query_fp = random.sample(aux_query_fp,1)
                rel_images_dict[q_label].extend(aux_query_fp)
            
            query_emb_to_be_searched = embeddings_query[query_idx].copy()
            labels = list(set(meta_data_test.class_name))
            tq = time.time()
            
            use_top_5_pred = 1
            perform_validation = 1
            fb_round = 0
            while fb_round<1:
                model = copy.deepcopy(model_copy)
                model = model.to(device)
                classifier = copy.deepcopy(classifier_copy)
                classifier = classifier.to(device)
                if fb_round==0:
                    use_top_5_pred = 0
                    perform_validation = 0
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
                        reviewed_indices = meta_data_search.copy().index[meta_data_search['file_path'].isin(reviewed_images_overall_copy)].tolist()
                        indices_prev = indices.copy().reshape(-1)
                        indices = np.array([x for x in list(indices_prev) if x not in list(reviewed_indices)])
                        prob_preds_sorted_search = prob_preds_search.reshape(-1)[indices.reshape(-1)]
                        gt = np.array(list(meta_data_search['class_name']))[indices]
                        to_be_reviewed_next = to_be_reviewed_next_based_on_pred(meta_data_search.copy(),indices,gt,prob_preds_sorted_search,q_label,num_samples=k-int(0.75*k))
                        relevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos]==q_label]
                        irrelevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos]!=q_label]
                        relevant.extend(relevant_prob_based)
                        irrelevant.extend(irrelevant_prob_based)

                    if args.only_classifier_pred:
                        relevant = []
                        irrelevant = []
                        reviewed_images_overall_copy = reviewed_images_overall+relevant+irrelevant
                        reviewed_indices = meta_data_search.index[meta_data_search['file_path'].isin(reviewed_images_overall_copy)].tolist()
                        indices_prev = indices.copy().reshape(-1)
                        indices = np.array([x for x in list(indices_prev) if x not in list(reviewed_indices)])
                        prob_preds_sorted_search = prob_preds_search.reshape(-1)[indices.reshape(-1)]
                        gt = np.array(list(meta_data_search['class_name']))[indices]
                        to_be_reviewed_next = to_be_reviewed_next_based_on_pred(meta_data_search,indices,gt,prob_preds_sorted_search,q_label,num_samples=k)
                        relevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos]==q_label]
                        irrelevant_prob_based = [i for i in to_be_reviewed_next if i.split('/')[-1*args.label_pos]!=q_label]
                        relevant.extend(relevant_prob_based)
                        irrelevant.extend(irrelevant_prob_based)
                    else:
                        relevant,irrelevant = get_rel_irrl(reviewed_images_overall,embeddings_search,meta_data_search,query_emb_to_be_searched,q_label,NN=k,strategy = args.ret_strategy,label_pos=args.label_pos)
                        
                
                relevant = list(set(relevant))
                irrelevant  = list(set(irrelevant))
                reviewed_images_overall.extend(relevant+irrelevant)
                reviewed_images_overall = list(set(reviewed_images_overall))
                for i in relevant+irrelevant:
                    class_label = i.split('/')[-2]
                    rel_images_dict[class_label].append(i)

                img_count = 0    
                for i in rel_images_dict.keys():
                    rel_images_dict[i] = list(set(rel_images_dict[i]))
                    img_count+=len(rel_images_dict[i])

                t1  =time.time()
                print("Rel in loop , Irr in loop and Img counts overall",fb_round,len(relevant),len(irrelevant),len(reviewed_images_overall),img_count,flush=True)
                model_updated ,last_epoch_metric_learning = update_model_metric_learning (model,args.search_dir,args.query_dir,args.annotated_dir,
                                                            rel_images_dict,args.label_set,fb_round,q_name)

                print('Metric learning updation done in ' , time.time()-t1,flush=True)
                
                embeddings_query,meta_data_query = get_emb_metadata_from_h5py_csv(model_updated,args.query_dir)
                embeddings_search,meta_data_search = get_emb_metadata_from_h5py_csv(model_updated,args.search_dir)
                embeddings_annotated,meta_data_annotated = get_emb_metadata_from_h5py_csv(model_updated,args.annotated_dir)
                
                
                query_emb_to_be_searched = embeddings_query[query_idx].copy()

                
                
                if args.use_query_refinement:
                    print('Using Query refinement')
                    query_emb_to_be_searched = get_updated_embeddings(query_emb_to_be_searched,rel_images_dict.copy()[q_label],meta_data_search,meta_data_annotated,meta_data_query,embeddings_search,embeddings_annotated,embeddings_query)    
                
            

                indices,preds = get_ranked_images(embeddings_search, query_emb_to_be_searched.reshape(-1,1),meta_data_search,leave_one_out=False)
                
                t3 = time.time()
                classifier_updated,last_epoch_classifier = run_classifier(classifier,args.search_dir,args.query_dir,args.annotated_dir,rel_images_dict,args.label_set,
                                                                           fb_round,q_name)
                
                print('Classifier training done in',time.time()-t3,flush=True)
                probs_query,meta_data_query = get_emb_metadata_from_h5py_csv(classifier_updated,args.query_dir,probs=1)
                probs_search, meta_data_search  = get_emb_metadata_from_h5py_csv(classifier_updated,args.search_dir,probs=1)
                probs_annot, meta_data_annot  = get_emb_metadata_from_h5py_csv(classifier_updated,args.annotated_dir,probs=1)
                
                prob_preds_search = np.array(args.label_set)[np.argmax(probs_search,axis=1)].reshape(-1)
                prob_preds_sorted_search = prob_preds_search.reshape(-1)[indices.reshape(-1)]
                fb_round+=1
                
            
            embeddings_test , meta_data_test  = get_emb_metadata_from_h5py_csv(model_updated,args.test_dir)
            probs_test, meta_data_test  = get_emb_metadata_from_h5py_csv(classifier_updated,args.test_dir,probs=1)
            prob_preds_test = np.array(args.label_set)[np.argmax(probs_test,axis=1)].reshape(-1)
            gt_test = np.array(list(meta_data_test.class_name))

            classes_for_reporting = args.label_set
            y_true = [classes_for_reporting.index(i) for i in gt_test]
            y_pred = [classes_for_reporting.index(i) for i in prob_preds_test]
            classification_res = classification_report(y_true, y_pred, target_names=classes_for_reporting, output_dict=True, digits=4)

            labels = list(set(meta_data_test.class_name))
            ret_df,ret_dict = get_ret_results_val(embeddings_search,embeddings_test,meta_data_search,meta_data_test,labels, 
                                                               K_for_P_at_K = 20,leave_one_out=False,N=1000,use_gpu_support=False,
                                                               save_sample_wise_results=True)  
            for i in args.label_set:
                final_csv_dict[i+'_Perfect_10'].append(ret_dict['Perfect_10'][i])
                

                final_csv_dict[i+'_P@20'].append(ret_dict['P@20'][i])
                
                final_csv_dict[i+'_P@50'].append(ret_dict['P@50'][i])
                
                final_csv_dict[i+'_P@100'].append(ret_dict['P@100'][i])
                
                final_csv_dict[i+'_MAP'].append(ret_dict['MAP'][i])
                

                final_csv_dict[i+'_Precision'].append(classification_res[i]['precision'])
                
                final_csv_dict[i+'_Recall'].append(classification_res[i]['recall'])
                
                final_csv_dict[i+'_F1'].append(classification_res[i]['f1-score'])
                
                final_csv_dict[i+'_num_samples_rev'].append(len(list(set(rel_images_dict[i]))))

            final_csv_dict['Overall_accuracy'].append(classification_res['accuracy'])    
            final_csv_dict['Perfect_10_average'].append(ret_dict['Perfect_10']['Micro'])
            pd.DataFrame(final_csv_dict).to_csv(os.path.join(args.save_prediction_csv,"Overall_test_set_inf.csv"))
            query_image_name = list(np.array(meta_data_query.file_path))[query_idx].split('/')[-1].split('.')[0]
            print('Time taken for query image {} is {}'.format(i,time.time()-tq),flush=True)
            torch.save(model_updated.state_dict(), "{}/metric_learning_model_{}.pth".format(args.model_save_path,query_image_name))
            torch.save(classifier_updated.state_dict(), "{}/classifier_model_{}.pth".format(args.model_save_path,query_image_name))
            use_top_5_pred = 1
            perform_validation = 1
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for Running Fb Retrieval from a search DB and a query DB")
    parser.add_argument("--save_results_name", type=str, default = "resenet18", help="method for metric learning")
    parser.add_argument("--search_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--query_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--test_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--annotated_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--annotated_img_dir", type=str, default = "./", help="method for metric learning")
    parser.add_argument("--num_img_to_rev", type=int, default = 50, help="method for metric learning")
    parser.add_argument("--use_aux_query", type=int, default = 0, help="method for metric learning")
    parser.add_argument("--entropy_based", type=int, default = 0, help="method for metric learning")
    parser.add_argument("--use_query_refinement", type=int, default=0)
    parser.add_argument("--ret_strategy", type=str, default="top_k_ret")
    parser.add_argument("--model", type=str, default="resnet-18")
    parser.add_argument("--save_prediction_csv", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="/ssd_scratch/cvit/ashishmenon/tensorboard_logs_ICIAR4/")
    parser.add_argument("--label_pos", type=int, default=2)
    parser.add_argument("--only_classifier_pred", type=int, default=0)
    parser.add_argument("--classifier_pred_top_mid_end", type=int, default=0)
    parser.add_argument("--use_resnet18", type=int, default=0)
    parser.add_argument("--label_set", type=str, nargs='+', default=None, help="Class Names")
    parser.add_argument("--model_save_path", type=str, default='./', help="Class Names")

    args = parser.parse_args()
    print(args,flush=True)
    os.makedirs(args.save_prediction_csv,exist_ok=True)
    os.makedirs(args.model_save_path,exist_ok=True)
    

    log_file = args.log_dir
    search_emb_files = glob.glob(args.search_dir+'/*.h5')
    search_emb_files.sort()
    search_annot_files = glob.glob(args.search_dir+'/*.csv')
    search_annot_files.sort()

    
    embeddings_search_list = []
    meta_data_search_list = []
    dataloader_search_list = []

    embeddings_test_list = []
    meta_data_test_list = []
    dataloader_test_list = []

    embeddings_val_list = []
    meta_data_val_list = []
    dataloader_val_list = []
    q_size = 0

    if args.label_set is not None:
        labels_database = args.label_set

    num_img_to_rev = args.num_img_to_rev


    
    resnet_model = models.resnet18(pretrained = True)
    resnet_model.fc = Identity()
    base_model_main = nn.Sequential(resnet_model.layer4[1],resnet_model.avgpool,nn.Flatten(),resnet_model.fc)

    resnet_model_classifier = models.resnet18(pretrained = True)
    resnet_model_classifier.fc = nn.Linear(512, 9)
    classifier_main  = nn.Sequential(resnet_model_classifier.layer4[1],resnet_model_classifier.avgpool,nn.Flatten(),resnet_model_classifier.fc)




    embeddings_query_512,meta_data_query = get_emb_metadata_from_h5py_csv(copy.deepcopy(base_model_main).to(device),args.query_dir)
        
    embeddings_search_512 , meta_data_search  = get_emb_metadata_from_h5py_csv(copy.deepcopy(base_model_main).to(device),args.search_dir)

    embeddings_annotated_512 , meta_data_annotated  = get_emb_metadata_from_h5py_csv(copy.deepcopy(base_model_main).to(device),args.annotated_dir)
    

    embeddings_test_512 , meta_data_test  = get_emb_metadata_from_h5py_csv(copy.deepcopy(base_model_main).to(device),args.test_dir)


    print("Loaded embeddings and annot done with search query val and held_out test sizes",len(embeddings_search_512),len(embeddings_query_512),len(embeddings_annotated_512),len(embeddings_test_512),flush=True)

    base_model = copy.deepcopy(base_model_main)
    classifier = copy.deepcopy(classifier_main)
    rel_images_dict = {}
    final_csv_dict = {}
    for i in args.label_set:
        final_csv_dict[i+'_P@20'] = []
        final_csv_dict[i+'_P@50'] = []
        final_csv_dict[i+'_P@100'] = []
        final_csv_dict[i+'_MAP'] = []
        final_csv_dict[i+'_Perfect_10'] = []
        final_csv_dict[i+'_Precision'] = []
        final_csv_dict[i+'_Recall'] = []
        final_csv_dict[i+'_F1'] = []
        final_csv_dict[i+'_num_samples_rev'] = []

    final_csv_dict['Perfect_10_average'] = []
    final_csv_dict['Overall_accuracy'] = []
    losses_classifier_dict_train = {}
    losses_classifier_dict_val = {}
    losses_metric_learning_dict_train = {}
    losses_metric_learning_dict_val = {}


    losses_metric_learning = SummaryWriter(log_dir=os.path.join(log_file,'losses_metric_learning'))
    
    losses_classifier = SummaryWriter(log_dir=os.path.join(log_file,'losses_classifier'))
    for i in glob.glob(args.annotated_img_dir+'/*/*.tif'):
        class_label = i.split('/')[-2]
        try:
            rel_images_dict[class_label].append(i)
        except:
            rel_images_dict[class_label] = []
            rel_images_dict[class_label].append(i)
    rel_fb_loop(base_model.to(device),classifier.to(device),embeddings_search_512, 
                embeddings_query_512,embeddings_annotated_512,embeddings_test_512,meta_data_search, 
                meta_data_query, meta_data_annotated, meta_data_test, rel_images_dict,k=num_img_to_rev,use_aux_query=args.use_aux_query)
    
       
        
