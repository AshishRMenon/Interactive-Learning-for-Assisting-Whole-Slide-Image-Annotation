import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import shutil

def get_reordered_dict(df,idx,map_dict = None , precision_dict = None ):
    df_dict = df.to_dict()
    if map_dict is None:
        map_dict  = collections.defaultdict(dict)

    if precision_dict is None:
        precision_dict  = collections.defaultdict(dict)

    for cl in list(df_dict['Class'].values()):
        map_dict['Step'][idx] = idx 
        precision_dict['Step'][idx] = idx 
        map_dict[cl][idx] = df[df['Class']==cl]['MAP'].item()
        precision_dict[cl][idx]= df[df['Class']==cl]['P_uniform'].item()
    return map_dict,precision_dict

def save_csv(df,name):
    with open('{}.csv'.format(name), "w") as f:
        pd.DataFrame(df).to_csv(f)

def get_rel_irr_stats(rel,irr,idx,state_dict=None):
    rel_classes_reviewed = [i.split('/')[-2] for i in rel]
    classes = list(set(rel_classes_reviewed))
    if state_dict is None:
        state_dict  = collections.defaultdict(dict)
    state_dict['Step'][idx] = idx
    state_dict['rel_classes_reviewed'][idx] = len(rel)
    state_dict['irrel_classes_reviewed'][idx] = len(irr)
    for cl in classes: 
        state_dict[cl][idx] = rel_classes_reviewed.count(cl)
    return state_dict
        

def move_file(fp):
    dest = fp.replace('not_reviewed' , 'reviewed')
    shutil.move(fp, dest)


def get_micro_macro_values(cl1,cl2):
    arr_value_cl1 = np.array(list(cl1.values()))
    arr_value_cl2 = np.array(list(cl2.values()))
    macro_value = arr_value_cl2.mean()
    micro_value = ((arr_value_cl1*arr_value_cl2)/(arr_value_cl1.sum())).sum()
    return macro_value, micro_value



def get_rel_irr_stats(rel,irr,idx,state_dict=None):
    rel_classes_reviewed = [i.split('/')[-2] for i in rel]
    classes = list(set(rel_classes_reviewed))
    if state_dict is None:
        state_dict  = collections.defaultdict(dict)
    state_dict['Step'][idx] = idx
    state_dict['rel_classes_reviewed'][idx] = len(rel)
    state_dict['irrel_classes_reviewed'][idx] = len(irr)
    for cl in classes: 
        state_dict[cl][idx] = rel_classes_reviewed.count(cl)
    return state_dict
