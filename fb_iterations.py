import numpy as np
from get_emb_ret import get_ranked_images,get_ret_results
import random
import math
def strip_samples_from_db(search_embeddings,search_annot,rev_images):
    indices = search_annot.index[search_annot['file_path'].isin(rev_images)].tolist()
    search_embeddings_copy = search_embeddings.copy()
    search_annot_copy = search_annot.copy()
    search_embeddings_copy = np.delete(search_embeddings_copy,indices,0)
    for i in indices: 
        search_annot_copy = search_annot_copy.drop([i])
    search_annot_copy = search_annot_copy.reset_index(drop=True)
    return search_embeddings_copy , search_annot_copy


def get_rel_irrl(reviewed_images,search_embeddings,search_annot,q_embedding,q_label,NN=50,strategy="top_k_ret",label_pos=2):
    search_embeddings_copy = search_embeddings.copy()
    search_annot_copy = search_annot.copy()
    search_embeddings_copy,search_annot_copy = strip_samples_from_db(search_embeddings_copy,search_annot_copy,reviewed_images)
    rel_fp , irr_fp = get_rel_irr_file_paths(search_embeddings_copy,q_embedding,search_annot_copy,q_label,num_neighbours=NN,strategy=strategy,label_pos=label_pos)
    return rel_fp,irr_fp

def get_rel_irr_file_paths(search_embeddings,query_embeddings,search_annot,query_label,num_neighbours=50,strategy="top_k_ret",label_pos=2):
    if strategy=='random_pick':  
        print("Using random sampling")
        file_paths = list(search_annot.file_path)
        random_samples = random.sample(file_paths, k=num_neighbours)
        if query_label=='Tumor':
            rel_img_paths = [i for i in random_samples if i.split('/')[-1*label_pos] in ['Benign','InSitu','Invasive','Tumor']]
            irr_img_paths = [i for i in random_samples if i.split('/')[-1*label_pos] not in ['Benign','InSitu','Invasive','Tumor']]
        else:
            rel_img_paths = [i for i in random_samples if i.split('/')[-1*label_pos] == query_label]
            irr_img_paths = [i for i in random_samples if i.split('/')[-1*label_pos] != query_label]
        
    else:
        file_paths = list(search_annot.file_path)
        indices_main,preds_main = get_ranked_images(search_embeddings, query_embeddings,search_annot,leave_one_out=False)
        indices_main = indices_main.reshape(-1)
        preds_main = preds_main.reshape(-1)
        if strategy == "top_k_ret":
            print("using top k ret")
            indices_ret = indices_main[:num_neighbours]
            predicted_ret = preds_main[:num_neighbours]
        
        if strategy == "front_mid_end_ret":
            num_front_samples = math.ceil(num_neighbours/3) 
            num_last_samples = math.ceil((num_neighbours-num_front_samples)/2)
            num_mid_samples = num_neighbours-num_front_samples-num_last_samples
            indices_front = indices_main[:num_front_samples]
            indices_mid = indices_main[(len(indices_main)//2)- (num_mid_samples//2): (len(indices_main)//2) + (num_mid_samples//2)+1]
            indices_last = indices_main[-num_last_samples:]
            indices_ret = np.concatenate((indices_front,indices_mid,indices_last))

            predicted_front = preds_main[:num_front_samples]
            predicted_mid = preds_main[(len(indices_main)//2) - (num_mid_samples//2): (len(indices_main)//2) + (num_mid_samples//2)+1]
            predicted_last = preds_main[-num_last_samples:]

            predicted_ret = np.concatenate((predicted_front,predicted_mid,predicted_last))

        relevance_fb = ((predicted_ret == query_label)*1).reshape(-1)
        relevant_indices = indices_ret[np.where(relevance_fb==1)[0]]
        irrelevant_indices = indices_ret[np.where(relevance_fb!=1)[0]]
        rel_img_paths = list(np.array(file_paths)[relevant_indices])
        irr_img_paths = list(np.array(file_paths)[irrelevant_indices])
    return rel_img_paths,irr_img_paths
