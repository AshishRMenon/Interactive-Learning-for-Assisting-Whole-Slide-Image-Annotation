from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import common_functions
import torch
import torch.nn as nn


def train_model_triplets(model, epoch , loss_func, mining_func, device, train_loader, val_loader, optimizer,perform_val=1):
    model.train()
    train_loss = 0
    val_loss = 0
    for batch_idx, (data, labels,fp,name) in enumerate(train_loader):
        # for i in range(len(labels)):    
        #     print(data[i],labels[i],fp[i])
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        # ref_label = labels.clone()
        # ref_label[ref_label!=torch.min(ref_label)]=1000
        # ref_label[ref_label==torch.min(labels)]=1
        # indices_tuple = mining_func(embeddings = out, ref_emb = out.clone(), labels = labels ,ref_labels=ref_label )
        indices_tuple = mining_func(out,labels)
        # indices_tuple_reqd = [[],[],[]]
        # print(len(indices_tuple[0]))
        # for i in range(len(indices_tuple[0])):
        #     if labels[indices_tuple[0][i]].item()==1 and labels[indices_tuple[1][i]].item()==1:
        #         indices_tuple_reqd[0].append(indices_tuple[0][i])
        #         indices_tuple_reqd[1].append(indices_tuple[1][i])
        #         indices_tuple_reqd[2].append(indices_tuple[2][i])
        num_hard_triplets = mining_func.num_triplets
        loss = loss_func(out, labels, indices_tuple)  
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        
        

    # print("Train Set : Epoch {} Iteration {}: Loss = {} num_hard_triplets = {}".format(epoch, batch_idx, train_loss,mining_func.num_triplets))
    if perform_val:
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels,fp,name) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)
                out = model(data)
                # ref_label = labels.clone()
                # ref_label[ref_label!=torch.min(ref_label)]=1000
                # ref_label[ref_label==torch.min(labels)]=1
                # indices_tuple = mining_func(embeddings = out, ref_emb = out.clone(), labels = labels ,ref_labels=ref_label )
                indices_tuple = mining_func(out,labels)
                num_hard_triplets = mining_func.num_triplets
                loss = loss_func(out, labels, indices_tuple)  
                val_loss+=loss.item()
    
    # print("VAL set , Epoch {} Iteration {}: Loss = {} num_hard_triplets = {}".format(epoch, batch_idx, val_loss,mining_func.num_triplets))
    return model,train_loss,val_loss

def train_classifier(model, epoch,device, train_loader, val_loader, optimizer,perform_val=1):
    model.train()
    train_loss = 0
    val_loss = 0
    for batch_idx, (data, labels,fp,name) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        # for i in range(len(labels)):
        #     print(labels[i],fp[i])
        optimizer.zero_grad()
        y_hat = model(data)
        loss = nn.CrossEntropyLoss()(y_hat,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    print("Train Set : Epoch {} Iteration {}: Loss = {} ".format(epoch, batch_idx, train_loss),flush=True)
    model.eval()
    if perform_val:
        with torch.no_grad():
            for batch_idx, (data, labels,fp,name) in enumerate(val_loader):
                # for i in range(len(labels)):    
                #     print(labels[i],fp[i])
                data, labels = data.to(device), labels.to(device)
                y_hat = model(data)
                loss = nn.CrossEntropyLoss()(y_hat,labels)
                val_loss+=loss.item()
    
    print("VAL set , Epoch {} Iteration {}: Loss = {} ".format(epoch, batch_idx, val_loss),flush=True)       
    return model,train_loss,val_loss


def train_classifier_cv(model, epoch,device, train_loader, val_loader, val_dataset,optimizer,perform_val=1):
    model.train()
    train_loss = 0
    val_loss = 0
    val_acc = 0
    for batch_idx, (data, labels,fp,name) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        y_hat = model(data)
        loss = nn.CrossEntropyLoss()(y_hat,labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    print("Train Set : Epoch {} Iteration {}: Loss = {} ".format(epoch, batch_idx, train_loss))
    model.eval()
    if perform_val:
        running_corrects = 0
        with torch.no_grad():
            for batch_idx, (data, labels,fp,name) in enumerate(val_loader):
                data, labels = data.to(device), labels.to(device)
                y_hat = model(data)
                _, preds = torch.max(y_hat, 1)
                loss = nn.CrossEntropyLoss()(y_hat,labels)
                val_loss+=loss.item()
                running_corrects += torch.sum(preds == labels.data)
    val_acc = running_corrects.double()/len(val_dataset)
    print("VAL set , Epoch {} Iteration {}: Loss = {} Accuracy = {} ".format(epoch, batch_idx, val_loss,val_acc))       
    return model,train_loss,val_loss,val_acc