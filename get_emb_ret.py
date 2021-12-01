import torch
import pandas as pd
import numpy as np
import faiss
import sklearn
from metrics import calculate_P_at_K, calculate_MAP,calculate_perfect_P_at_K
from utils import get_micro_macro_values
from sklearn.metrics import classification_report
from torch import nn
import torch.nn.functional as F
import torch
import time
import h5py
import torch.nn as nn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
def get_model_embeddings_from_dataloader(net,dataloader_in,probs=0,classifier_loss=0):  
    net = net.cuda()
    # net = nn.DataParallel(net).cuda()
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_in):
            img_data = data[0].cuda()
            out = net(img_data)
            if classifier_loss:
                loss = list(nn.CrossEntropyLoss(reduction="none")(out,data[1].cuda()).cpu().numpy())
            else:
                loss = [0]*len(data[0])
            if probs:
                prob = F.softmax(out,dim=1)
                try:
                    probabilities = np.concatenate((probabilities,prob.detach().cpu().numpy().copy()),axis=0)
                except:
                    probabilities = prob.detach().cpu().numpy().copy()
            else:
                try:
                    embeddings = np.concatenate((embeddings,out.detach().cpu().numpy().copy()),axis=0)
                except:
                    embeddings = out.detach().cpu().numpy().copy()                    
            d_class_id = pd.DataFrame(list(data[1]))
            d_file_path = pd.DataFrame(list(data[2]))
            d_class_name = pd.DataFrame(list(data[3]))
            d_loss_per_img  = pd.DataFrame(loss)
            d_combined = pd.concat([d_class_id,d_file_path,d_class_name,d_loss_per_img], axis=1)
            try:
                total_database = total_database.append(d_combined)

            except:
                total_database = d_combined
            del img_data
            del out
            if probs:
                del prob
            torch.cuda.empty_cache()
    df = {'class_id': list(total_database.iloc[:,0]), 'file_path': list(total_database.iloc[:,1]) , 'class_name': list(total_database.iloc[:,2]),'Loss': list(total_database.iloc[:,3])}
    dataframe = pd.DataFrame(df)
    if probs:
        return probabilities,dataframe
    else:
        return embeddings,dataframe



def get_model_embeddings_from_dataloader_cpu(net,dataloader_in,probs=0):  
    net.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_in):
            img_data = data[0]
            out = net(img_data)
            prob = F.softmax(out,dim=1)
            try:
                embeddings = np.concatenate((embeddings,out.cpu().numpy().copy()),axis=0)
                probabilities = np.concatenate((probabilities,prob.cpu().numpy().copy()),axis=0)
            except:
                embeddings = out.cpu().numpy().copy()
                probabilities = prob.cpu().numpy().copy()
                    
            d_class_id = pd.DataFrame(list(data[1]))
            d_file_path = pd.DataFrame(list(data[2]))
            d_class_name = pd.DataFrame(list(data[3]))
            d_combined = pd.concat([d_class_id,d_file_path,d_class_name], axis=1)
            try:
                total_database = total_database.append(d_combined)

            except:
                total_database = d_combined
    df = {'class_id': list(total_database.iloc[:,0]), 'file_path': list(total_database.iloc[:,1]) , 'class_name': list(total_database.iloc[:,2])}
    dataframe = pd.DataFrame(df)
    if probs:
        return probabilities,dataframe
    else:
        return embeddings,dataframe

def get_model_embeddings_from_embeddings(net,embeddings,device):  
    embeddings = torch.from_numpy(embeddings).to(device)
    embeddings = embeddings.view(1,embeddings.shape[0],embeddings.shape[1])
    net.eval()
    with torch.no_grad():
        out= net(embeddings)
    out = out.cpu().numpy()
    out = out.reshape(out.shape[1],out.shape[2])
    return out




def get_ranked_images(search_embeddings, query_embeddings, search_annot,query_annot=None, knn_algo='brute',  knn_metric='euclidean', \
                  K_for_P_at_K = 5 , use_faiss=True,use_gpu_support=False,n_neighbors=None,leave_one_out=True,get_id=False,get_distances=False):
    if n_neighbors:
        n_neighbors = n_neighbors
    else:
        n_neighbors= len(search_embeddings)
    if use_faiss:
        index_flat = faiss.IndexFlatL2(search_embeddings.shape[1])
        if use_gpu_support:
            res = faiss.StandardGpuResources()
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(search_embeddings)         # add vectors to the index
            t1 = time.time()
            try:
                D, indices_near = gpu_index_flat.search(query_embeddings, min(1000, n_neighbors))  # actual search
            except:
                query_embeddings = query_embeddings.reshape(1,-1)
                D, indices_near = gpu_index_flat.search(query_embeddings, min(1000, n_neighbors))
        else:
            index_flat.add(search_embeddings)
            t1 = time.time()               
            try:
                D, indices_near = index_flat.search(query_embeddings, n_neighbors)  # actual search
            except:
                query_embeddings = query_embeddings.reshape(1,-1)
                D, indices_near = index_flat.search(query_embeddings, n_neighbors)
                
        t2 = time.time()
    else:
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm=knn_algo,metric=knn_metric)
        t1 = time.time()
        neighbors.fit(search_embeddings)
        distances, indices_near = neighbors.kneighbors(query_embeddings)
        t2 = time.time()
    
    indices_near = indices_near.astype(int)
    distances_near = D
    if leave_one_out:
        indices_near = indices_near[:,1:]
        distances_near = D[:,1:]

    pred_labels = np.array(search_annot.class_name)[indices_near]
    pred_id = np.array(search_annot.class_id)[indices_near]
    if get_distances:
        if get_id:
            return indices_near,pred_labels,pred_id,distances_near
        else:
            return indices_near,pred_labels,distances_near
    
    else:
        if get_id:
            return indices_near,pred_labels,pred_id
        else:
            return indices_near,pred_labels
        



def get_ret_results(search_embeddings,query_embeddings,search_annot,query_annot, label_mapping,K_for_P_at_K = 100,leave_one_out=True,N=1000,
                    use_gpu_support=False,save_sample_wise_results=False,val_only=False):
    total_class_list = list(set(query_annot.class_name))
    total_class_list.sort()
    class_names = total_class_list
    num_classes = len(class_names)
    class_names.sort()
    precision_weighted_dict = {}
    precision_uniform_dict = {}
    num_samples = {} 
    precision_dict_5 = {}
    precision_dict_10 = {}
    precision_dict_20 = {}
    precision_dict_50 = {}
    precision_dict_100 = {}
    MAP = {}
    class_info = {}
    MAP_at_K = []
    P_at_K = []
    pr = {}
    map_val = {}
    for cnt,gt_class_name in enumerate(label_mapping):
        gt_class_id = total_class_list.index(gt_class_name)
        query_embeddings_classwise = query_embeddings[query_annot.class_name==gt_class_name]    
        _,predicted_classes_array,pred_id = get_ranked_images(search_embeddings,query_embeddings_classwise,search_annot,
                                                      leave_one_out=leave_one_out,n_neighbors=N,use_gpu_support=use_gpu_support,get_id=True)
        gt_classes_array = np.array(query_annot[query_annot.class_name == gt_class_name].class_name)
        result = (gt_classes_array.reshape(-1,1)==predicted_classes_array)
        
        
        Precision_5 = calculate_MAP(result,K=5).mean()
        Precision_10 = calculate_MAP(result,K=10).mean()
        Precision_20 = calculate_MAP(result,K=20).mean()
        Precision_50 = calculate_MAP(result,K=50).mean()
        Precision_100 = calculate_MAP(result,K=100).mean()

        precision_dict_5[gt_class_name] = Precision_5
        precision_dict_10[gt_class_name] = Precision_10
        precision_dict_20[gt_class_name] = Precision_20
        precision_dict_50[gt_class_name] = Precision_50
        precision_dict_100[gt_class_name] = Precision_100
        num_samples[gt_class_name] = len(gt_classes_array)

        for k,i in enumerate(label_mapping):
            gt_class_to_check = np.array([i]*len(query_embeddings_classwise))

            result_check = (gt_class_to_check.reshape(-1,1)==predicted_classes_array)
            if k==0:
                pr[cnt] = calculate_P_at_K(result_check,5)["P_at_K_sample_wise"].reshape(-1,1)
                map_val[cnt]  = calculate_MAP(result_check,K=5).reshape(-1,1)
            else:
                pr[cnt] = np.concatenate((pr[cnt],calculate_P_at_K(result_check,5)["P_at_K_sample_wise"].reshape(-1,1)),axis=1)
                map_val[cnt] = np.concatenate((map_val[cnt],calculate_MAP(result_check,K=5).reshape(-1,1)),axis=1)
        if cnt==0:
            y_true = gt_classes_array.copy()
        else:
            y_true = np.concatenate((y_true,gt_classes_array)) 
    
    for ck in pr.keys():
        try:
            final_pr = np.concatenate((final_pr,pr[ck]))
            final_map = np.concatenate((final_map,map_val[ck]))
        except:
            final_pr = pr[ck].copy()
            final_map = map_val[ck].copy()
    
    y_pred_P5 = np.array(label_mapping)[np.argmax(final_pr,axis=1)]
    y_pred_AP5 = np.array(label_mapping)[np.argmax(final_map,axis=1)]    

    if val_only:
        meta_data = search_annot.copy()    
        meta_data['Final_pred_P@5'] =  y_pred_P5
        meta_data['Final_pred_MAP'] =  y_pred_AP5

    else:
        meta_data = query_annot.copy()
        meta_data['Final_pred_P@5'] =  y_pred_P5
        meta_data['Final_pred_MAP'] =  y_pred_AP5

        meta_data_search = search_annot.copy()
        meta_data_search['Final_pred_P@5'] =  meta_data_search['class_name']
        meta_data_search['Final_pred_MAP'] =  meta_data_search['class_name']
        meta_data = meta_data.append(meta_data_search.copy(), ignore_index=True)

    report_P5 = classification_report(y_true, y_pred_P5, target_names=label_mapping, output_dict=True, digits=4)
    report_AP5 = classification_report(y_true, y_pred_AP5, target_names=label_mapping, output_dict=True, digits=4)


    precision_dict_5['Macro'] , precision_dict_5['Micro'] = get_micro_macro_values(num_samples,precision_dict_5)
    precision_dict_10['Macro'] , precision_dict_10['Micro'] = get_micro_macro_values(num_samples,precision_dict_10)
    precision_dict_20['Macro'] , precision_dict_20['Micro'] = get_micro_macro_values(num_samples,precision_dict_20)
    precision_dict_50['Macro'] , precision_dict_50['Micro'] = get_micro_macro_values(num_samples,precision_dict_50)
    precision_dict_100['Macro'] , precision_dict_100['Micro'] = get_micro_macro_values(num_samples,precision_dict_100)
    num_samples['Macro'] = '-'
    num_samples['Micro'] = '-'
    # class_info['Macro'] = 'Macro'
    # class_info['Micro'] = 'Micro'
    
    
    final_dict = {'Class': class_info , 'NUM_SAMPLES':num_samples,'AP@5':precision_dict_5, 'AP@10':precision_dict_10, 'AP@20':precision_dict_20,
                 'AP@50':precision_dict_50,'AP@100':precision_dict_100,'MAP':MAP
                 }


    df = pd.DataFrame(final_dict)    
    if save_sample_wise_results:
        return meta_data,report_P5,report_AP5
    # else:
    #     return df


# def get_ret_results(search_embeddings,query_embeddings,search_annot,query_annot, label_mapping,K_for_P_at_K = 100,leave_one_out=True,N=1000,use_gpu_support=False,save_sample_wise_results=False,save_name='final_pred'):
#     total_class_list = list(set(query_annot.class_name))
#     total_class_list.sort()
#     class_names = total_class_list
#     num_classes = len(class_names)
#     class_names.sort()
#     precision_weighted_dict = {}
#     precision_uniform_dict = {}
#     num_samples = {} 
#     precision_dict_5 = {}
#     precision_dict_10 = {}
#     precision_dict_20 = {}
#     precision_dict_50 = {}
#     precision_dict_100 = {}
#     MAP = {}
#     class_info = {}
#     MAP_at_K = []
#     P_at_K = []
#     for cnt,gt_class_name in enumerate(class_names):
#         gt_class_id = total_class_list.index(gt_class_name)
#         query_embeddings_classwise = query_embeddings[query_annot.class_name==gt_class_name]    
#         _,predicted_classes_array,pred_id = get_ranked_images(search_embeddings,query_embeddings_classwise,search_annot,
#                                                       leave_one_out=leave_one_out,n_neighbors=N,use_gpu_support=use_gpu_support,get_id=True)
#         gt_classes_array = np.array(query_annot[query_annot.class_name == gt_class_name].class_name)
#         result = (gt_classes_array.reshape(-1,1)==predicted_classes_array)
        
        
#         Precision_5 = calculate_MAP(result,K=5).mean()
#         Precision_10 = calculate_MAP(result,K=10).mean()
#         Precision_20 = calculate_MAP(result,K=20).mean()
#         Precision_50 = calculate_MAP(result,K=50).mean()
#         Precision_100 = calculate_MAP(result,K=100).mean()

#         precision_dict_5[gt_class_name] = Precision_5
#         precision_dict_10[gt_class_name] = Precision_10
#         precision_dict_20[gt_class_name] = Precision_20
#         precision_dict_50[gt_class_name] = Precision_50
#         precision_dict_100[gt_class_name] = Precision_100
#         num_samples[gt_class_name] = len(gt_classes_array)

#         masked_annot = query_annot[query_annot['class_name']==gt_class_name]

#         pr = None   
#         map_val = None 
#         for k,i in enumerate(label_mapping):
#             gt_class_to_check = np.array([i]*len(query_embeddings_classwise))

#             result_check = (gt_class_to_check.reshape(-1,1)==predicted_classes_array)
#             if k!=0:
#                 pr = np.concatenate((pr,calculate_P_at_K(result_check,5)["P_at_K_sample_wise"].reshape(-1,1)),axis=1)
#                 map_val = np.concatenate((map_val,calculate_MAP(result_check,K=5).reshape(-1,1)),axis=1)
#             if k==0:
#                 pr = calculate_P_at_K(result_check,5)["P_at_K_sample_wise"].reshape(-1,1)
#                 map_val  = calculate_MAP(result_check,K=5).reshape(-1,1)
#         masked_annot["Final_pred_P@5"] = np.array(label_mapping)[np.argmax(pr,axis=1)]                
#         masked_annot["Final_pred_MAP"] = np.array(label_mapping)[np.argmax(map_val,axis=1)]
#         if cnt==0:
#             meta_data = masked_annot.copy()
#         if cnt!=0:
#             meta_data = meta_data.append(masked_annot.copy(), ignore_index=True)
    

#     masked_annot = search_annot.copy()    
#     masked_annot['Final_pred_P@5'] =  masked_annot['class_name']
#     masked_annot['Final_pred_MAP'] =  masked_annot['class_name']
    
#     meta_data = meta_data.append(masked_annot.copy(), ignore_index=True)
    
#     y_true = np.array(list(meta_data['class_name']))
#     y_pred_P5 = np.array(list(meta_data['Final_pred_P@5']))
#     y_pred_AP5 = np.array(list(meta_data['Final_pred_MAP']))
#     classes = label_mapping
    
#     report_P5 = classification_report(y_true, y_pred_P5, target_names=classes, output_dict=True, digits=4)
#     report_AP5 = classification_report(y_true, y_pred_AP5, target_names=classes, output_dict=True, digits=4)


#     precision_dict_5['Macro'] , precision_dict_5['Micro'] = get_micro_macro_values(num_samples,precision_dict_5)
#     precision_dict_10['Macro'] , precision_dict_10['Micro'] = get_micro_macro_values(num_samples,precision_dict_10)
#     precision_dict_20['Macro'] , precision_dict_20['Micro'] = get_micro_macro_values(num_samples,precision_dict_20)
#     precision_dict_50['Macro'] , precision_dict_50['Micro'] = get_micro_macro_values(num_samples,precision_dict_50)
#     precision_dict_100['Macro'] , precision_dict_100['Micro'] = get_micro_macro_values(num_samples,precision_dict_100)
#     num_samples['Macro'] = '-'
#     num_samples['Micro'] = '-'
#     # class_info['Macro'] = 'Macro'
#     # class_info['Micro'] = 'Micro'
    
    
#     final_dict = {'Class': class_info , 'NUM_SAMPLES':num_samples,'AP@5':precision_dict_5, 'AP@10':precision_dict_10, 'AP@20':precision_dict_20,
#                  'AP@50':precision_dict_50,'AP@100':precision_dict_100,'MAP':MAP
#                  }


#     df = pd.DataFrame(final_dict)    
#     if save_sample_wise_results:
#         return meta_data,report_P5,report_AP5
#     # else:
#     #     return df



def get_ret_results_val(search_embeddings,query_embeddings,search_annot,query_annot, label_mapping,K_for_P_at_K = 100,leave_one_out=True,N=1000,use_gpu_support=False,save_sample_wise_results=False,save_name='final_pred'):
    total_class_list = list(set(query_annot.class_name))
    total_class_list.sort()
    class_names = total_class_list
    num_classes = len(class_names)
    class_names.sort()
    precision_weighted_dict = {}
    precision_uniform_dict = {}
    num_samples = {} 
    precision_dict_5 = {}
    precision_dict_10 = {}
    precision_dict_20 = {}
    precision_dict_50 = {}
    precision_dict_100 = {}
    precision_perfect_dict_10 = {}
    MAP_dict = {}
    class_info = {}
    MAP_at_K = []
    P_at_K = []
    for cnt,gt_class_name in enumerate(class_names):
        gt_class_id = total_class_list.index(gt_class_name)
        query_embeddings_classwise = query_embeddings[query_annot.class_name==gt_class_name]    
        _,predicted_classes_array,pred_id = get_ranked_images(search_embeddings,query_embeddings_classwise,search_annot,
                                                      leave_one_out=False,n_neighbors=N,use_gpu_support=use_gpu_support,get_id=True)
        gt_classes_array = np.array(query_annot[query_annot.class_name == gt_class_name].class_name)
        result = (gt_classes_array.reshape(-1,1)==predicted_classes_array)
        
        
        Precision_5 = calculate_P_at_K(result,K=5)['P_at_K_uniform']
        Precision_10 = calculate_P_at_K(result,K=10)['P_at_K_uniform']
        Precision_20 = calculate_P_at_K(result,K=20)['P_at_K_uniform']
        Precision_50 = calculate_P_at_K(result,K=50)['P_at_K_uniform']
        Precision_100 = calculate_P_at_K(result,K=100)['P_at_K_uniform'] 
        Precision_perfect_10 = calculate_perfect_P_at_K(result,K=10)

        precision_dict_5[gt_class_name] = Precision_5
        precision_dict_10[gt_class_name] = Precision_10
        precision_dict_20[gt_class_name] = Precision_20
        precision_dict_50[gt_class_name] = Precision_50
        precision_dict_100[gt_class_name] = Precision_100
        precision_perfect_dict_10[gt_class_name] = Precision_perfect_10
        MAP_dict[gt_class_name]= calculate_MAP(result).mean()
        num_samples[gt_class_name] = len(gt_classes_array)
        

    precision_dict_5['Macro'] , precision_dict_5['Micro'] = get_micro_macro_values(num_samples,precision_dict_5)
    precision_dict_10['Macro'] , precision_dict_10['Micro'] = get_micro_macro_values(num_samples,precision_dict_10)
    precision_dict_20['Macro'] , precision_dict_20['Micro'] = get_micro_macro_values(num_samples,precision_dict_20)
    precision_dict_50['Macro'] , precision_dict_50['Micro'] = get_micro_macro_values(num_samples,precision_dict_50)
    precision_dict_100['Macro'] , precision_dict_100['Micro'] = get_micro_macro_values(num_samples,precision_dict_100)
    precision_perfect_dict_10['Macro'] , precision_perfect_dict_10['Micro'] = get_micro_macro_values(num_samples,precision_perfect_dict_10)
    MAP_dict['Macro'] , MAP_dict['Micro'] = get_micro_macro_values(num_samples,MAP_dict)
    num_samples['Macro'] = '-'
    num_samples['Micro'] = '-'
    class_info['Macro'] = 'Macro'
    class_info['Micro'] = 'Micro'
    
    
    final_dict = {'Class': class_info , 'NUM_SAMPLES':num_samples,'P@5':precision_dict_5, 'P@10':precision_dict_10, 'P@20':precision_dict_20,
                 'P@50':precision_dict_50,'P@100':precision_dict_100, 'Perfect_10':precision_perfect_dict_10,'MAP':MAP_dict
                 }


    df = pd.DataFrame(final_dict)    
    return df,final_dict