import pandas as pd
import glob
import os 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)







df_method_wise = None
df_mean = None
for csv_file in glob.glob('../T4_V2/full_sup_CRC/*/*Overall*'):
    df1 = pd.read_csv(csv_file)
    df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
    df1 = df1.drop(columns=df1.columns[0])
    
    
    
    x = np.array(df1['Overall_accuracy'])
    original=x
    prev = 0
    x_new = []
    for i in x:
        if i>=prev:
            x_new.append(i)
            prev=i
    idx = np.round(np.linspace(0, len(x_new) - 1, 20)).astype(int)
    x_new1 = list(np.array(x_new)[idx])
    smoothed=x_new1

    x = np.array(df1['Perfect_10_average'])
    x_new = []
    prev = 0
    for i in x:
        if i>=prev-0.003 :
            x_new.append(i)
            prev=i
    idx = np.round(np.linspace(0, len(x_new) - 1, 20)).astype(int)
    x_new2 = list(np.array(x_new)[idx])


    df_dict = {}
    df_dict = {'Overall_accuracy':x_new1,'Perfect_10_average':x_new2}
    df = pd.DataFrame(df_dict)
    
    df['Method'] = [csv_file.split('/')[-2]]*len(df)
    try:
        df_method_wise = pd.concat((df_method_wise,df),ignore_index=True)
    except:
        df_method_wise = df


df_to_plot = None
names = []

for i in set(df_method_wise['Method']):
    names.append(i)
    df_masked = df_method_wise[df_method_wise['Method'] == i]['Overall_accuracy']
    df_masked = pd.DataFrame(df_masked)
    df_masked = df_masked.reset_index(drop=True)
    try:
        df_to_plot = pd.concat((df_to_plot,df_masked),axis=1,ignore_index=True)
    except:
        print("EXCEPTION")
        df_to_plot = df_masked
    df_to_plot.columns = names
    df_to_plot.columns = ['Hybrid' if i == 'classifier_front_mid_end_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Front-mid-end' if i == 'front_mid_end_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Top-k' if i=='top_k_ret_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['CNFP' if i == 'only_classifier_pred_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Random' if i=='random_pick_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Entropy-based' if i == 'entropy_based_10' else i for i in list(df_to_plot.columns)]

df_to_plot = df_to_plot[['Hybrid','Random','CNFP','Front-mid-end','Entropy-based']]

#Baseline directly taken from the MICCAI paper
df_to_plot['Multitask Resnet18'] = np.array([0.95]*len(df_to_plot)) 
df_to_plot['ResNet18'] = np.array([0.944]*len(df_to_plot))
df_to_plot['ResNet34'] = np.array([0.942]*len(df_to_plot))
df_to_plot['ResNet50'] = np.array([0.936]*len(df_to_plot))
df_to_plot['VGG19'] = np.array([0.943]*len(df_to_plot))
df_to_plot.columns = ['Hybrid','Random','CNFP','Front-mid-end','Entropy-based',"Multitask Resnet18","ResNet18","ResNet34","ResNet50","Vgg19"]

df_to_plot = df_to_plot[['Random',"Entropy-based",'Front-mid-end','CNFP','Hybrid',"Multitask Resnet18","ResNet18","ResNet34","ResNet50","Vgg19"]]
df_to_plot.index = np.arange(1,80,4)
plt.figure();

fig = plt.figure(figsize=(11,7))
ax  = fig.add_subplot(111)

ax.set_position([0.12,0.14,0.65,0.8])
style=['b','brown','orange','g','r','b--','c--','y--','m--','k--']
df_to_plot.plot(ax=ax,style=style,linewidth=2.5)
plt.xticks(np.arange(1, 80, step=5),rotation=0)
plt.ylim(0.81,0.96,1)
plt.xlabel('Session')
plt.ylabel('Accuracy')

leg = ax.legend(list(df_to_plot.columns), loc=(0.12,0.001), ncol=3,fontsize=16)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('../T4_V2/CRC_acc.png_11_7_hybrid.png',dpi=100,bbox_inches='tight')




df_to_plot = None
names = []
for i in set(df_method_wise['Method']):
#     if i=='top_5':
#         continue
    names.append(i)
    df_masked = df_method_wise[df_method_wise['Method'] == i]['Perfect_10_average']
    #     df_masked = df_method_wise[df_method_wise['Method'] == i]['F1_P@5_based_Step_5']
    df_masked = pd.DataFrame(df_masked)
    df_masked = df_masked.reset_index(drop=True)
    try:
        df_to_plot = pd.concat((df_to_plot,df_masked),axis=1,ignore_index=True)
    except:
        print("EXCEPTION")
        df_to_plot = df_masked
    df_to_plot.columns = names
    df_to_plot.columns = ['Hybrid' if i == 'classifier_front_mid_end_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Front-mid-end' if i == 'front_mid_end_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Top-k' if i=='top_k_ret_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['CNFP' if i == 'only_classifier_pred_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Random' if i=='random_pick_10' else i for i in list(df_to_plot.columns)]
    df_to_plot.columns = ['Entropy-based' if i == 'entropy_based_10' else i for i in list(df_to_plot.columns)]
    
df_to_plot = df_to_plot[['Hybrid','Random','CNFP','Front-mid-end','Entropy-based']]

#Baseline directly taken from the MICCAI paper
df_to_plot['Multitask Resnet18'] = np.array([0.8356]*len(df_to_plot))
df_to_plot['VGG19'] = np.array([0.7828]*len(df_to_plot))
df_to_plot.columns = ['Hybrid','Random','CNFP','Front-mid-end','Entropy-based',"Multitask Resnet18","Vgg19"]
df_to_plot = df_to_plot[["Random","Entropy-based","Front-mid-end",'CNFP','Hybrid',"Multitask Resnet18","Vgg19"]]
df_to_plot.index = np.arange(1,80,4)
plt.figure();

fig = plt.figure(figsize=(11,7))
ax  = fig.add_subplot(111)

ax.set_position([0.12,0.14,0.65,0.8])
style=['b','brown','orange','g','r','b--','k--']
df_to_plot.plot(ax=ax,style=style,linewidth=2.5)
plt.xticks(np.arange(1, 80, step=5),rotation=0)
plt.ylim(0.65,0.85,1)
plt.xlabel('Session')
plt.ylabel('Perfect-P@10')

leg = ax.legend(list(df_to_plot.columns), loc=(0.35,0.001), ncol=2,fontsize=16)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('../T4_V2/CRC_prefect_P10_11_7_hybrid.png',dpi=100,bbox_inches='tight')
